#ifndef Y3SPACE_DRIVER_YOST_STATS_H_
#define Y3SPACE_DRIVER_YOST_STATS_H_


#include <super_lib/am_stat_list.h>
#include <super_lib/am_stat.h>
#include <super_lib/am_stat_reset.h>
#include <super_lib/am_stat_ave.h>
#include <super_lib/am_stat_status.h>

namespace am
{
class YostStats
{
public:
    AMStatStatus statStatus = AMStatStatus("ss", "AMStatStatus");
    AMStatReset imu_pub = AMStatReset("imu_freq", "IMU publish Hz", 100, 150, 350, 375);

    YostStats(AMStatList &list)
    {
        list.add(&statStatus);
        list.add(&imu_pub);
    }

};
}

#endif /*Y3SPACE_DRIVER_YOST_STATS_H_*/