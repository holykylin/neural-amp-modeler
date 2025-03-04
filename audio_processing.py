def load_audio(file_path, sr=44100):
    # 修改前
    audio, _ = librosa.load(file_path, sr=sr, mono=True)  # 强制单声道
    
    # 修改后
    audio, _ = librosa.load(file_path, sr=sr, mono=False)  # 保持原始声道数
    if len(audio.shape) == 1:
        audio = np.expand_dims(audio, axis=0)  # 单声道转二维
    return audio.transpose(1, 0)  # 转换为 (samples, channels) 