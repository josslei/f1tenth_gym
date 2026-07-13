from controllers.controller_base import ControlCommand, Controller, VehicleState

class LMPCController(Controller):
    def __init__(self) -> None:
        self.vehicle_state = VehicleState(0.0, 0.0, 0.0, 0.0)

    def reset(self) -> None:
        self.vehicle_state = VehicleState(0.0, 0.0, 0.0, 0.0)

    def update(self, vehicle_state: VehicleState) -> None:
        self.vehicle_state = vehicle_state
        pass

    def control(self) -> ControlCommand:
        return ControlCommand(0, 0)