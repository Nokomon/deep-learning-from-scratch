import numpy as np

class Modify:
    def __init__(self, W):
        self.params = [W]

W = np.random.randn(3, 4)
result = [Modify(W) for _ in range(6)]
result = result[:]

for i in range(len(result)-1):
    for j in range(i+1, len(result)):
        try:
            # print(result[i] == result[j])
            print(result[i] is result[j])
        except:
            print("A")


def remove_duplicate(params, grads):
    '''
    매개변수 배열 중 중복되는 가중치를 하나로 모아
    그 가중치에 대응하는 기울기를 더한다.
    '''
    params, grads = params[:], grads[:]  # copy list

    while True:
        find_flg = False
        L = len(params)

        for i in range(0, L - 1):
            for j in range(i + 1, L):
                # 가중치 공유 시
                if params[i] is params[j]:
                    print(i, j, "matches")
                    grads[i] += grads[j]  # i인덱스에 경사를 더함
                    find_flg = True
                    params.pop(j)
                    grads.pop(j)
                # 가중치를 전치행렬로 공유하는 경우(weight tying)
                elif params[i].ndim == 2 and params[j].ndim == 2 and \
                     params[i].T.shape == params[j].shape and np.all(params[i].T == params[j]):
                    grads[i] += grads[j].T
                    find_flg = True
                    params.pop(j)
                    grads.pop(j)

                if find_flg: break
            if find_flg: break

        if not find_flg: break

    return params, grads