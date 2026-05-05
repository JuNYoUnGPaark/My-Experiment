import numpy as np
from scipy.signal import butter, filtfilt, resample
from scipy.interpolate import interp1d


# =============================================================================
# Gaussian noise
# =============================================================================
def add_gaussian_noise(x, sigma=0.005, seed=42):
		# 난수 생성
    rng = np.random.default_rng(seed)  
  
    # 노이즈 생성
    noise = rng.normal(loc=0.0, scale=sigma, size=x.shape)  
    return x + noise  # 신호에 더함(additive)


def add_gaussian_noise_channelwise(x, level=0.2, seed=42):
    rng = np.random.default_rng(seed)
    x = x.astype(np.float32)

    # 채널별 표준편차 계산 
    sigma = x.std(axis=0) + 1e-8

    # noise 생성 
    noise = rng.normal(loc=0.0, scale=1.0, size=x.shape).astype(np.float32)

    # 채널별 scaling 적용
    noise = noise * (level * sigma)

    return (x + noise).astype(np.float32)


def add_gaussian_jitter(x, sigma=0.01, seed=42):
    return add_gaussian_noise(x, sigma=sigma, seed=seed)

  
# =============================================================================
# Impulse noise (Spike) 
# =============================================================================
def add_impulse_noise(x, spike_ratio=0.03, spike_scale=0.5, seed=42):
    rng = np.random.default_rng(seed)  # 난수 생성
    x_noisy = x.copy()  # 미리 변형신호를 만들기 위해 copy 

    T = x.shape[0]
    # Spike 개수 설정 (T = 전체 데이터 개수, Batch 따로 X)
    num_spikes = int(T * spike_ratio)
		
		# Spike 발생 위치 선택
    spike_indices = rng.choice(T * C, size=num_spikes, replace=False)  
    
    # Spike 값 생성 (가우시안 기반)
    spike_values = rng.normal(loc=0.0, scale=spike_scale, size=num_spikes)

		# Spike 적용 
    flat[spike_indices] += spike_values  

    return x_noisy, spike_indices

# =============================================================================
# Bias drift
# =============================================================================
def add_bias_drift(x, alpha=0.05):
    T = x.shape[0]
    
    # 시간 축 생성 
    # 0과 1사이의 T개의 수를 선형 적으로 증가/감소하도록 생성 
    # [0.0, 0.01, 0.02, ..., 1.0] 
    t = np.linspace(0, 1, T).reshape(T, 1)
    
    # drift 값 생성
    drift = alpha * t
		
    return x + drift  # 원신호에 drift 적용 
    
    
# 또다른 방법: 채널별로 다른 방향으로 drift를 적용해서 현실성을 높일 수도 있다.     
def add_bias_drift_channelwise(x, alpha=0.05):
    T, C = x.shape

    t = np.linspace(0, 1, T).reshape(T, 1)

    # 채널별 drift 방향 다르게
    drift_direction = np.random.uniform(-1, 1, size=(1, C))
    drift = alpha * t * drift_direction

    return x + drift


# =============================================================================
# Gravity shift (Baseline shift)
# =============================================================================
def add_gravity_shift(x, shift=0.1, channels=None):
    x_shifted = x.copy()
		
		# 전체 채널에 적용 
    if channels is None:
        x_shifted += shift
    # 특정 채널만 적용 
    else:
        x_shifted[:, channels] += shift

    return x_shifted


# =============================================================================
# Linear drift
# =============================================================================
def add_linear_drift(x, start=0.0, end=0.2):
    T = x.shape[0]
    
    # 시작~끝을 정한다. (나머지는 2-1. Bias drift와 동일)
    drift = np.linspace(start, end, T).reshape(T, 1)

    return x + drift


# =============================================================================
# Scale  drift
# =============================================================================
def scale_drift(x, start=1.0, end=2.5):
    T = x.shape[0]
    
    # 시간에 따라 변하는 scaling factor 생성
    alpha = np.linspace(start, end, T).reshape(T, 1)

    return x * alpha  # 신호를 시간별로 다르게 scaling 

def scale_drift_channelwise(x, start=1.0, end=1.5):
    T, C = x.shape
    alpha = np.linspace(start, end, T).reshape(T, 1)
    direction = np.random.uniform(0.8, 1.2, size=(1, C))
    return x * alpha * direction   


# =============================================================================
# Magnitude scaling
# =============================================================================
def magnitude_scaling(x, alpha=1.2):
    return x * alpha


# =============================================================================
# Sensor group scaling
# =============================================================================
def sensor_group_scaling(x, group_indices, scale_factors):
    x_scaled = x.copy()

    for group, alpha in zip(group_indices, scale_factors):
        x_scaled[:, group] *= alpha

    return x_scaled


# =============================================================================
# Clipping / Saturation
# =============================================================================
def clip_signal(x, threshold=1.0):
    x_clipped = np.clip(x, -threshold, threshold)
    return x_clipped


# =============================================================================
# Temporal scaling
# =============================================================================
def temporal_scaling_keep_length(x, alpha=1.2):
    T, C = x.shape
		
		# 원래의 시간축을 0~1까지 정규화
    original_time = np.linspace(0, 1, T)
    
    # 변형된 시간축 생성 
    scaled_time = np.linspace(0, 1, T) * alpha

    # 범위를 벗어나는 부분 방지
    scaled_time = np.clip(scaled_time, 0, 1)
    
    # 결과 array 생성 
    x_scaled = np.zeros_like(x)

    for c in range(C):
        f = interp1d(  # 채널별로 interpolation
            original_time,
            x[:, c],
            kind="linear",
            fill_value="extrapolate"
        )
        # 새로운 시간축으로 샘플링 
        x_scaled[:, c] = f(scaled_time)

    return x_scaled


# =============================================================================
# Resampling
# =============================================================================
def resampling_perturbation_keep_length(x, scale=0.8):
    T, C = x.shape

    intermediate_T = max(2, int(T * scale))

    # 1. 길이 변경
    x_resampled = resample(x, intermediate_T, axis=0)

    # 2. 다시 원래 길이로 복원
    x_back = resample(x_resampled, T, axis=0)

    return x_back.astype(np.float32)


# =============================================================================
# Local time distortion (Time warping)
# =============================================================================
def time_warping(x, sigma=0.2, knot=4, seed=42):
    rng = np.random.default_rng(seed)
    T, C = x.shape
    
    # 원래 시간: [0, 1, 2, ..., T-1]
    original_time = np.arange(T)

    # 시간축을 몇 개 구간으로 나눔
    # knot + 2개 기준점 생성: (시작점, 중간 knot들, 끝점)
    knot_x = np.linspace(0, T - 1, knot + 2)

    # 각 기준점에서 구간별 속도 생성
    random_warp = rng.normal(loc=1.0, scale=sigma, size=knot + 2)

    # 음수 또는 너무 작은 값 방지
    random_warp = np.clip(random_warp, 0.1, None)

    # 기준점 사이를 선형 보간해서 구간별 속도를 부드럽게 연결
    warp_curve = np.interp(original_time, knot_x, random_warp)

    # 누적합으로 새로운 시간축 생성
    warped_time = np.cumsum(warp_curve)

    # 0 ~ T-1 범위로 정규화
    warped_time = (warped_time- warped_time[0]) / (warped_time[-1] - warped_time[0])
    warped_time = warped_time * (T - 1)

    x_warped = np.zeros_like(x)
		
		# Interpolation
    for c in range(C):
        f = interp1d(
            original_time,
            x[:, c],
            kind="linear",
            fill_value="extrapolate"
        )
        x_warped[:, c] = f(warped_time)

    return x_warped.astype(np.float32)


# =============================================================================
# Window shifting
# =============================================================================
def window_shift_padding(x, shift=10, mode="edge"):
    T, C = x.shape
    shifted = np.zeros_like(x)

		# padding 방식 선택 
    if mode == "edge":  # 가장자리 값 유지
        pad_value_start = x[0:1]
        pad_value_end = x[-1:]
    elif mode == "zero":  # 0으로 채움
        pad_value_start = np.zeros((1, C), dtype=x.dtype)
        pad_value_end = np.zeros((1, C), dtype=x.dtype)
    elif mode == "mean":  # 평균값으로 채움 
        mean_value = np.mean(x, axis=0, keepdims=True)
        pad_value_start = mean_value
        pad_value_end = mean_value
    else:
        raise ValueError("mode must be one of ['edge', 'zero', 'mean']")

		# 오른쪽 shift
    if shift > 0:
        shifted[shift:] = x[:-shift]
        shifted[:shift] = pad_value_start
		
		# 왼쪽 shift
    elif shift < 0:
        shift_abs = abs(shift)
        shifted[:-shift_abs] = x[shift_abs:]
        shifted[-shift_abs:] = pad_value_end

		# shift가 없을 때
    else:
        shifted = x.copy()

    return shifted


# =============================================================================
# Rotation / Orientation 변화
# =============================================================================
# 1. x/y/z축 기준 회전 
def rotation_matrix_x(theta):
    c, s = np.cos(theta), np.sin(theta)
		
		# x축은 그대로, y-z 평면만 회전시킨다. 
    return np.array([
        [1, 0, 0],
        [0, c, -s],
        [0, s,  c]
    ], dtype=np.float32)

def rotation_matrix_y(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([
        [ c, 0, s],
        [ 0, 1, 0],
        [-s, 0, c]
    ], dtype=np.float32)


def rotation_matrix_z(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([
        [c, -s, 0],
        [s,  c, 0],
        [0,  0, 1]
    ], dtype=np.float32)  


# 2. Euler angle 기반 3D rotation matrix를 이용하여 회전 순서 정의 
def euler_rotation_matrix(rx=0.0, ry=0.0, rz=0.0, degrees=True):
    # degree -> radian 변환 
    if degrees:
        rx, ry, rz = np.deg2rad([rx, ry, rz])
		
		# 각 축 회전 정의 
    Rx = rotation_matrix_x(rx)
    Ry = rotation_matrix_y(ry)
    Rz = rotation_matrix_z(rz)

    # 최종 회전: 회전 순서 = Z -> Y -> X
    R = Rz @ Ry @ Rx

    return R.astype(np.float32)
    
    
# 3. 적용 
def apply_rotation_3axis(x_3axis, R):
    return (x_3axis @ R.T).astype(np.float32)


# =============================================================================
# Permutation
# =============================================================================
def axis_permutation(x_3axis, order=(1, 0, 2)):
    return x_3axis[:, order].astype(np.float32)


# =============================================================================
# Sign flip
# =============================================================================
def sign_flip(x_3axis, signs=(-1, 1, 1)):
    signs = np.array(signs).reshape(1, 3)
    return (x_3axis * signs).astype(np.float32)


# =============================================================================
# Channel Dropout
# =============================================================================
def channel_dropout(x, drop_prob=0.3, seed=42):
    # 난수 생성 
    rng = np.random.default_rng(seed)

    T, C = x.shape
		
		# 채널축을 기준으로 살릴지 결정 
		# mask 생성: 0~1사이의 랜덤값이고, drop_prob보다 크면 True, 작으면 False 
    mask = rng.random(C) > drop_prob  
    
    # float로 변환 
    mask = mask.astype(np.float32)

    return x * mask  # dropout 적용 


# =============================================================================
# Temporal Dropout
# =============================================================================
def temporal_dropout(x, drop_prob=0.1, seed=42):
		# 난수 생성
    rng = np.random.default_rng(seed)

    T, C = x.shape
		
		# 시간축을 기준으로 살릴지(drop) 결정
    mask = rng.random(T) > drop_prob 
    mask = mask.astype(np.float32).reshape(T, 1)

    return x * mask


# =============================================================================
# Frequency masking
# =============================================================================
def frequency_masking(x, fs=50.0, f_low=0.5, f_high=2.0):
    T, C = x.shape

    x_masked = np.zeros_like(x, dtype=np.float32)

    for c in range(C):
        signal = x[:, c]

        # FFT
        X_f = np.fft.rfft(signal)
        # FFT는 index만 주기 때문에 실제 Hz로 변환
        freqs = np.fft.rfftfreq(T, d=1/fs)  

        # mask 생성 (제거할 구간 = 0)
        mask = np.ones_like(freqs)
        mask[(freqs >= f_low) & (freqs <= f_high)] = 0  # 특정 대역만 0으로 만든다.

        # masking 적용
        X_f_masked = X_f * mask

        # inverse FFT: 다시 time-domain으로 복원 
        x_masked[:, c] = np.fft.irfft(X_f_masked, n=T)

    return x_masked






