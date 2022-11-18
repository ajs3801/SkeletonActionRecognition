from configs import actions


class Counter:
    def __init__(self, count_threshold):
        self.cur_flags = {
            "squat-down": False,
            "squat-up": False,
            "pushup-down": False,
            "pushup-up": False,
            "lunge-down": False,
            "lunge-up": False,
            "stand": False,
            "push2stand": False,
            "stand2push": False,
        }
        self.prev_flags = {
            "squat-down": False,
            "squat-up": False,
            "pushup-down": False,
            "pushup-up": False,
            "lunge-down": False,
            "lunge-up": False,
            "stand": False,
            "push2stand": False,
            "stand2push": False,
        }
        self.cnt = {"squat": 0, "pushup": 0, "lunge": 0}
        self.threshold = count_threshold

    # 카운트 초기화
    def reset_cnt(self):
        for key, val in self.cnt.items():
            self.cnt[key] = 0

    # 이전 플래그 전부 false로 초기화
    def reset_prev_flags(self):
        for action in actions:
            self.prev_flags[action] = False

    # 현재 플래그 전부 false로 초기화
    def reset_cur_flags(self):
        for action in actions:
            self.cur_flags[action] = False

    # 현재 플래그가 켜져있는 액션을 반환
    def get_cur_flag(self):
        for key, val in self.cur_flags.items():
            if val:
                return key

    # 현재 플래그에서 입력으로 받은 액션만 키고 나머지는 다 끔
    def cur_flag_on(self, action):
        self.reset_cur_flags()
        self.cur_flags[action] = True

    # 이전 플래그에서 입력으로 받은 액션만 키고 나머지는 다 끔
    def prev_flag_on(self, action):
        self.reset_prev_flags()
        self.prev_flags[action] = True

    # 현재 플래그가 켜져있는 운동을 이전 플래그로 옮김
    def cur2prev(self):
        cur_flag = self.get_cur_flag()
        self.reset_cur_flags()
        self.reset_prev_flags()
        self.prev_flag_on(cur_flag)

    # 입력 : action window
    # 출력 : 카운트된 운동
    def count(self, action_window):
        for action in actions:
            if action_window.count(action) >= self.threshold:
                self.cur2prev()
                self.cur_flag_on(action)

        if self.prev_flags["squat-down"] and self.cur_flags["squat-up"]:
            self.cnt["squat"] += 1
            return "S"
        elif self.prev_flags["pushup-down"] and self.cur_flags["pushup-up"]:
            self.cnt["pushup"] += 1
            return "P"
        elif self.prev_flags["lunge-down"] and self.cur_flags["lunge-up"]:
            self.cnt["lunge"] += 1
            return "L"
        else:
            return None
