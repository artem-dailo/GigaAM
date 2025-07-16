from gigaam.onnx_utils import load_onnx_sessions, recognise_emotion
onnx_dir=""
model_type = "emo"
wav_file = "path_to_wav_file"

if __name__ == '__main__':

    sessions = load_onnx_sessions(onnx_dir, model_type, model_version="v1")
    out = recognise_emotion(wav_file, sessions)
    print(out)