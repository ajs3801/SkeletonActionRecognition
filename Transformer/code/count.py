from configs import *

action_count_length = config['action_count_length']

cur_flags = {'squat-down': False, 'squat-up': False, 'pushup-down': False,
             'pushup-up': False, 'lunge-down': False, 'lunge-up': False,
             'stand': False, 'push2stand': False, 'stand2push': False}

prev_flags = {'squat-down': False, 'squat-up': False, 'pushup-down': False,
              'pushup-up': False, 'lunge-down': False, 'lunge-up': False,
              'stand': False, 'push2stand': False, 'stand2push': False}

cnt = {'squat': 0, 'pushup': 0, 'lunge': 0}


def resetCnt():
    for key, val in cnt.items():
        cnt[key] = 0


def resetPrevFlags():  # 이전 플래그 전부 false로 초기화
    for action in actions:
        prev_flags[action] = False


def resetCurFlags():  # 현재 플래그 전부 false로 초기화
    for action in actions:
        cur_flags[action] = False


def getCurFlags():  # 현재 플래그가 켜져있는 액션을 반환
    for key, val in cur_flags.items():
        if val:
            return key


def curFlagOn(action):  # 입력으로 받은 액션의 현재 플래그만 키고 나머지는 다 끔
    resetCurFlags()
    cur_flags[action] = True


def prevFlagOn(action):  # 입력으로 받은 액션의 이전 플래그만 키고 나머지는 다 끔
    resetPrevFlags()
    prev_flags[action] = True


def cur2prev():  # 현재 플래그가 켜져있는 운동을 찾아서 이전 플래그에서 그 운동을 킴
    cur_flag = getCurFlags()
    resetCurFlags()
    resetPrevFlags()
    prevFlagOn(cur_flag)


def countAction(action_count):
    for a in actions:
        if action_count.count(a) >= 6:
            cur2prev()
            curFlagOn(a)

    if prev_flags['squat-down'] and cur_flags['squat-up']:
        cnt['squat'] += 1
        return True, 'S'
    elif prev_flags['pushup-down'] and cur_flags['pushup-up']:
        cnt['pushup'] += 1
        return True, 'P'
    elif prev_flags['lunge-down'] and cur_flags['lunge-up']:
        cnt['lunge'] += 1
        return True, 'L'
    else:
        return False, None
