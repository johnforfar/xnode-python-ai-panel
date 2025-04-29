"use client";

import { MediaRecorder, register } from "extendable-media-recorder";
import { connect } from "extendable-media-recorder-wav-encoder";
import { useEffect, useState } from "react";

export function useMediaRecorder() {
  const [recorder, setRecorder] = useState<MediaRecorder | undefined>(
    undefined
  );

  useEffect(() => {
    connect()
      .then(register)
      .then(() =>
        navigator.mediaDevices.getUserMedia({
          audio: {
            sampleRate: 24000,
            sampleSize: 16,
            channelCount: 1,
          },
        })
      )
      .then((stream) => {
        setRecorder(
          new MediaRecorder(stream, {
            mimeType: "audio/wav",
          }) as MediaRecorder
        );
      });
  }, []);

  return recorder;
}
