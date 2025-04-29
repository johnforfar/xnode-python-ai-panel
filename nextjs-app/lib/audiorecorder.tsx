"use client";

export class AudioRecorder {
  private audioContext = new AudioContext({ sampleRate: 24000 });
  private recording = false;
  private recorder: MediaRecorder | undefined;
  private onAudio: ((audio: Float32Array) => void) | undefined;
  private onStop: (() => void) | undefined;

  public async init({
    onAudio,
    onStop,
  }: {
    onAudio?: (audio: Float32Array) => void;
    onStop?: () => void;
  }) {
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: { sampleRate: 24000, channelCount: 1 },
    });
    this.recorder = new MediaRecorder(stream);
    if (this.recorder.state === "recording") {
      this.recorder.stop();
    }
    this.onAudio = onAudio;
    this.onStop = onStop;

    const microphone = this.audioContext.createMediaStreamSource(stream);

    const processorNode = this.audioContext.createScriptProcessor(4096, 1, 1);
    microphone.connect(processorNode);
    processorNode.connect(this.audioContext.destination);

    processorNode.onaudioprocess = (e) => {
      if (!this.recording) {
        return;
      }

      this.onAudio?.(e.inputBuffer.getChannelData(0));
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
    this.recorder?.start();
  }

  public stop() {
    this.recording = false;
    this.recorder?.stop();
    this.onStop?.();
  }

  public update({
    onAudio,
    onStop,
  }: {
    onAudio?: (audio: Float32Array) => void;
    onStop?: () => void;
  }) {
    this.onAudio = onAudio;
    this.onStop = onStop;
  }
}
