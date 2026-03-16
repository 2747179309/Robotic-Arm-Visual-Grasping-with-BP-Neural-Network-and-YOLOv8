import numpy as np
from scipy.io import loadmat

def tansig(x):
    # MATLAB tansig: 2/(1+exp(-2x)) - 1
    return 2.0 / (1.0 + np.exp(-2.0 * x)) - 1.0

def logsig(x):
    return 1.0 / (1.0 + np.exp(-x))

def purelin(x):
    return x

_ACT = {
    "tansig": tansig,
    "logsig": logsig,
    "purelin": purelin,
}

def _to_str(t):
    """把 tf 字段稳健转成 'tansig'/'purelin' 这样的字符串"""
    if isinstance(t, bytes):
        return t.decode("utf-8")
    t = str(t)
    # 处理 "b'tansig'" 这种情况
    if t.startswith("b'") and t.endswith("'"):
        t = t[2:-1]
    return t

def mapminmax_apply(x, ps):
    # x: (D, M)
    xoffset = np.array(ps["xoffset"], dtype=float).reshape(-1, 1)
    gain    = np.array(ps["gain"], dtype=float).reshape(-1, 1)
    ymin    = float(np.array(ps["ymin"], dtype=float).reshape(()))
    return (x - xoffset) * gain + ymin

def mapminmax_reverse(y, ps):
    xoffset = np.array(ps["xoffset"], dtype=float).reshape(-1, 1)
    gain    = np.array(ps["gain"], dtype=float).reshape(-1, 1)
    ymin    = float(np.array(ps["ymin"], dtype=float).reshape(()))
    return (y - ymin) / gain + xoffset

class PitchPredictor:
    def __init__(self, mat_path="pitch_model_py.mat"):
        # simplify_cells=True 会把 MATLAB struct/cell 尽量转成 dict/list
        mat = loadmat(mat_path, squeeze_me=True, struct_as_record=False, simplify_cells=True)
        model = mat["model"]  # 这里通常是 dict

        if not isinstance(model, dict):
            raise TypeError(f"期望 model 是 dict，但得到 {type(model)}。请检查 .mat 导出格式。")

        # dict 方式取字段
        self.W  = model["W"]   # list
        self.b  = model["b"]   # list
        self.tf = model["tf"]  # list
        self.psX = model["psX"]  # dict
        self.psY = model["psY"]  # dict

        # 转成 numpy
        self.W = [np.array(w, dtype=float) for w in self.W]
        self.b = [np.array(b, dtype=float).reshape(-1, 1) for b in self.b]
        self.tf = [_to_str(t) for t in self.tf]

    def predict(self, x, y, z=20.0):
        # 支持标量或数组
        if y < 0:
            y = -y
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)

        if np.isscalar(z):
            z = np.full_like(x, float(z), dtype=float)
        else:
            z = np.asarray(z, dtype=float)

        if x.shape != y.shape or x.shape != z.shape:
            raise ValueError("x,y,z 形状必须一致（或 z 为标量）。")

        # (3, M)
        Xin = np.vstack([x.ravel(), y.ravel(), z.ravel()])

        # 输入归一化
        Xn = mapminmax_apply(Xin, self.psX)

        # 前向传播
        a = Xn
        for Wi, bi, tfi in zip(self.W, self.b, self.tf):
            f = _ACT.get(tfi)
            if f is None:
                raise RuntimeError(f"不支持的激活函数: {tfi}，支持：{list(_ACT.keys())}")
            a = f(Wi @ a + bi)

        # 输出反归一化
        Y = mapminmax_reverse(a, self.psY)  # (1, M)

        return Y.reshape(x.shape)

if __name__ == "__main__":
    pred = PitchPredictor("pitch_model_py.mat")
    p = pred.predict(12, 12)  # 默认 z=20
    print("pitch =", p)
