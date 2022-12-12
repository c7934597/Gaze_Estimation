from .item import Item

class DataVerify:
    def pre_check_data(data: Item) -> bool:
        values = data.values
        length = len(values)
        if length == 15:
            return True
        else:
            return False

    def check_zero_data(y_head_pitch, y_head_yaw) -> bool:
        valueToBeRemoved = 0
        y_head_pitch = [value for value in y_head_pitch if value != valueToBeRemoved]
        y_head_yaw = [value for value in y_head_yaw if value != valueToBeRemoved]
        if len(y_head_pitch) < 8 or len(y_head_yaw) < 8:
            return True
        else:
            return False
