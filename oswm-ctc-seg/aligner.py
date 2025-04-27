import soundfile as sf
from espnet2.bin.s2t_ctc_align import CTCSegmentation


def get_aligner(lang_sym="<eng>"):
    aligner = CTCSegmentation(
        s2t_model_file="owsm_ctc_v3.2_ft_1B/exp/s2t_train_s2t_multitask-ctc_ebf27_conv2d8_size1024_raw_bpe50000/valid.total_count.ave_5best.till45epoch.pth",
        fs=16000,
        ngpu=1,
        batch_size=16,    # batched parallel decoding; reduce it if your GPU memory is smaller
        kaldi_style_text=True,
        time_stamps="fixed",
        samples_to_frames_ratio=1280,   # 80ms time shift; don't change as it depends on the pre-trained model
        lang_sym=lang_sym,
        task_sym="<asr>",
        context_len_in_secs=2,  # left and right context in buffered decoding
        frames_per_sec=12.5,    # 80ms time shift; don't change as it depends on the pre-trained model
    )
    return aligner



