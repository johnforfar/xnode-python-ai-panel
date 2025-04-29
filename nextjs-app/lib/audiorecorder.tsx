"use client";

export class AudioRecorder {
  private audioContext = new AudioContext({ sampleRate: 24000 });
  private recording = false;
  private recorder: MediaRecorder | undefined;

  public async init({ onAudio }: { onAudio: (audio: Float32Array) => void }) {
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: { sampleRate: 24000, channelCount: 1 },
    });
    this.recorder = new MediaRecorder(stream);
    console.log(this.recorder.audioBitsPerSecond);

    const microphone = this.audioContext.createMediaStreamSource(stream);

    const processorNode = this.audioContext.createScriptProcessor(4096, 1, 1);
    microphone.connect(processorNode);
    processorNode.connect(this.audioContext.destination);

    processorNode.onaudioprocess = (e) => {
      if (!this.recording) {
        return;
      }

      onAudio(e.inputBuffer.getChannelData(0));
    };
  }

  public getRecorder() {
    return this.recorder;
  }

  public isRecording() {
    return this.recording;
  }

  public start() {
    this.recording = true;
  }

  public stop() {
    this.recording = false;
  }
}
