import rclpy
import csv
from parrot_drone.drone_setup  import DroneAnafi
from rclpy.node import Node
from  geometry_msgs.msg import Vector3
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy




class FlyDrone(Node):    
    def __init__(self):
        super().__init__('fly_drone')

        qos = QoSProfile(
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=100,
        )

        self.publisher = self.create_publisher(Vector3, '/gps_targets', qos)

        self.declare_parameter("test_num", f"0")
        test_num = self.get_parameter("test_num").value
        self.csv_path = f"/home/drl/Data/target_detection/test{test_num}.csv"
        self._next_id = 0
        self._init_csv()


        self.parrot = DroneAnafi(2,25,10)
        self.planner = self.parrot.flight_planner

    def _init_csv(self):
        try:
            with open(self.csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["id", "drone_lat", "drone_lon", "target_lat", "target_lon"])
        except OSError as e:
            self.get_logger().error(f"Failed to initialize CSV: {e}")

    def _log_detection(self, drone_lat, drone_lon, target_lat, target_lon):
        row = (self._next_id, drone_lat, drone_lon, target_lat, target_lon)
        self._next_id += 1
        try:
            with open(self.csv_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(row)
        except OSError as e:
            self.get_logger().error(f"Failed to write CSV row: {e}")


    def execute_scan(self):
            #scan assumes you are starting from right most and bottom most portion of the grid and working left
        dir_counter = -1
        rows, col = self.planner.generate_flight_grid()

        #You will move the drone forward to take a picture at each row in a column, once that column is complete
        # drone will rotate, and shift to the next column. 

        self.parrot.takeoff()

        for column in range(col):
            for row in range(rows):
                try:
                    if self.parrot.move_straight(self.parrot.coverage):
                        self.get_logger().info('Taking RGB image')

                        result, pos = self.parrot.take_photo_rgb()
                        lat,lon,_ = self.parrot.get_gps()

                        self.get_logger().info(f"Drone photo at {lat}, {lon} ")

                        if result:
                            for position in pos:
                                self.get_logger().info(f"Target found at {position[0]}, {position[1]} ")
                                self._log_detection(lat, lon, position[0], position[1])
                                msg = Vector3()
                                msg.x = position[0]
                                msg.y = position[1]
                                msg.z = 0.0
                                self.publisher.publish(msg)

                except: 
                    self.get_logger().error('Failed scan step')


            
            try:
                    mod = dir_counter ** column 
                    self.parrot.rotate()
                    self.parrot.move_right(mod*self.parrot.coverage)
                    
            except Exception as e: 
                self.logger.error(f'Failed scan step: {e}')

    def land(self):
        self.parrot.land()

    def test_network(self):
        result, pos = self.parrot.take_photo_rgb()

        if result:
            for position in pos:
                self.get_logger().info(f"Target found at {position[0]}, {position[1]} ")
                msg = Vector3()
                msg.x = position[0]
                msg.y = position[1]
                msg.z = 0.0
                self.publisher.publish(msg)


def main(args=None):
    rclpy.init(args=args)

    fly_drone = FlyDrone()


    fly_drone.execute_scan()
    fly_drone.land()
    #fly_drone.test_network()
    fly_drone.parrot.disconnect()

    fly_drone.destroy_node()
    rclpy.shutdown()
