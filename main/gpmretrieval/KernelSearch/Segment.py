import gpbasics.global_parameters as global_param

global_param.ensure_init()

from gpbasics.Statistics import GaussianProcess as gp

global next_id
next_id = 0


class Segment:
    def __init__(self):
        global next_id
        self.id: int = next_id
        next_id += 1

        self.final: bool = False
        self.latest_metric: float = None
        self.latest_gp: gp.AbstractGaussianProcess = None

    def get_latest_metric(self, use_auxiliary_values: bool = True) -> float:
        if self.is_final() and use_auxiliary_values:
            return 1e+399
        return self.latest_metric

    def is_final(self) -> bool:
        return self.final

    def set_final(self, final):
        if final:
            self.block_data_input = None
        self.final = final

    def set_latest_metric(self, latest_metric: float):
        self.latest_metric = latest_metric

    def get_latest_gp(self) -> gp.AbstractGaussianProcess:
        return self.latest_gp

    def set_latest_gp(self, latest_gp: gp.AbstractGaussianProcess):
        self.latest_gp = latest_gp


class LegacySegment(Segment):
    def __init__(self, start_index: int, stop_index: int):
        super(LegacySegment, self).__init__()
        self.start_index: int = start_index
        self.stop_index: int = stop_index

    def get_start_index(self) -> int:
        return self.start_index

    def get_stop_index(self) -> int:
        return self.stop_index
