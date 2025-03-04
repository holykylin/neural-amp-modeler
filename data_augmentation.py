def random_time_shift(audio, max_shift=100):
    # 修改前（单声道处理）
    shift = np.random.randint(-max_shift, max_shift)
    return np.roll(audio, shift, axis=0)
    
    # 修改后（多声道处理）
    shift = np.random.randint(-max_shift, max_shift)
    return np.roll(audio, shift, axis=1)  # 在时间维度上统一平移 