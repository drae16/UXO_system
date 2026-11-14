#include <y3space_driver/yost_node.h>

namespace am
{
YostNode::YostNode(const std::string &node_name) : AMLifeCycle(node_name)
{

}

void YostNode::setClass(std::shared_ptr<Y3SpaceDriver> am_class)
{
    yost_class = am_class;
}

std::shared_ptr<Y3SpaceDriver> YostNode::getClass()
{
    return yost_class;
}

void YostNode::heartbeatCB()
{
    if(!yost_class->is_open())
    {
        AMLifeCycle::errorTerminal("Serial Port is down","SR0");
    }
    AMLifeCycle::heartbeatCB();
}

bool YostNode::onCleanup()
{
    ROS_INFO("onCleanup");
    
    yost_class->onCleanup();
    return AMLifeCycle::onCleanup();
}

bool YostNode::onConfigure()
{
    if(configured_)
    {
        return AMLifeCycle::onConfigure();
    }

    ROS_INFO("onConfigure");

    if(!yost_class->onConfigure())
    {
        ROS_WARN("yost_class->onConfigure() failed");
        yost_class->onCleanup();
        return false;
    }
    else
    {
        configured_ = true;
        return AMLifeCycle::onConfigure();
    }
}
}