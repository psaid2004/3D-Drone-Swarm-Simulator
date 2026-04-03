# -*- coding: utf-8 -*-
class Dummy:
    def __init__(self, min_val, max_val, speed):
        self.min = min_val
        self.max = max_val
        self.speed = speed

    def position_at_time(self, t):
        path_length = self.max - self.min
        period = 2 * path_length / self.speed
        t_in_cycle = t % period
        if t_in_cycle < period / 2:
            pos = self.min + self.speed * t_in_cycle
        else:
            pos = self.max - self.speed * (t_in_cycle - period / 2)
        return pos


obj = Dummy(2, 10, 2)

for t in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
    print(f"t={t}, pos={obj.position_at_time(t)}")
