from resemblyzer import VoiceEncoder, preprocess_wav
import numpy as np
import io
import librosa
import streamlit as st


# stricter threshold
VOICE_THRESHOLD = 0.82

# minimum gap between top match and second match
VOICE_MARGIN = 0.05


@st.cache_resource
def load_voice_encoder():
    return VoiceEncoder()


def get_voice_embedding(audio_bytes):
    try:
        encoder = load_voice_encoder()

        audio_buffer = io.BytesIO(audio_bytes)

        audio, sr = librosa.load(
            audio_buffer,
            sr=16000,
            mono=True,
        )

        wav = preprocess_wav(audio)

        embedding = encoder.embed_utterance(wav)

        return embedding.tolist()

    except Exception as e:
        st.error(f"Voice recognition error: {e}")
        return None


def identify_speaker(
    new_embedding,
    candidates_dict,
    threshold=VOICE_THRESHOLD,
    margin=VOICE_MARGIN,
):
    if new_embedding is None or not candidates_dict:
        return None, 0.0

    scores = []

    # normalize new embedding
    new_emb = np.array(new_embedding)
    new_emb = new_emb / (np.linalg.norm(new_emb) + 1e-9)

    for sid, stored_embedding in candidates_dict.items():
        if not stored_embedding:
            continue

        stored_emb = np.array(stored_embedding)
        stored_emb = stored_emb / (
            np.linalg.norm(stored_emb) + 1e-9
        )

        similarity = np.dot(new_emb, stored_emb)

        scores.append((sid, similarity))

    if not scores:
        return None, 0.0

    # sort highest similarity first
    scores.sort(key=lambda x: x[1], reverse=True)

    best_sid, best_score = scores[0]

    second_score = scores[1][1] if len(scores) > 1 else -1

    # debug while testing
    # st.write(scores)

    if (
        best_score >= threshold
        and (best_score - second_score) >= margin
    ):
        return best_sid, best_score

    return None, best_score


def process_bulk_audio(
    audio_bytes,
    candidates_dict,
    threshold=VOICE_THRESHOLD,
):
    try:
        encoder = load_voice_encoder()

        audio_buffer = io.BytesIO(audio_bytes)

        audio, sr = librosa.load(
            audio_buffer,
            sr=16000,
            mono=True,
        )

        # split speech sections
        segments = librosa.effects.split(
            audio,
            top_db=25,
        )

        identified_results = {}

        for start, end in segments:

            # ignore tiny clips
            if (end - start) < sr * 1.0:
                continue

            segment_audio = audio[start:end]

            wav = preprocess_wav(segment_audio)

            embedding = encoder.embed_utterance(wav)

            sid, score = identify_speaker(
                embedding,
                candidates_dict,
                threshold,
            )

            if sid:
                if (
                    sid not in identified_results
                    or score > identified_results[sid]
                ):
                    identified_results[sid] = score

        return identified_results

    except Exception as e:
        st.error(f"Bulk process error: {e}")
        return {}