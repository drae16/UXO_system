#ifndef Y3SPACE_DRIVER_YOST_NODE_H_
#define Y3SPACE_DRIVER_YOST_NODE_H_

#include <y3space_driver/Y3SpaceDriver.h>
#include <super_lib/am_life_cycle.h>

namespace am
{
class YostNode : public AMLifeCycle
{
public:
    std::shared_ptr<Y3SpaceDriver> yost_class = nullptr;

    YostNode(const std::string &node_name);

    void setClass(std::shared_ptr<Y3SpaceDriver> am_class);

    std::shared_ptr<Y3SpaceDriver> getClass();

    bool configured_ = false;

    void heartbeatCB() override;

    bool onCleanup() override;

    bool onConfigure() override;
};
}

#endif