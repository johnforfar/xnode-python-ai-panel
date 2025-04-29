"use client";

export class AudioRecorder {
  private audioContext = new AudioContext({ sampleRate: 24000 });
  private recorder: MediaRecorder | undefined;

  public async init({ onAudio }: { onAudio: (audio: Float32Array) => void }) {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    this.recorder = new MediaRecorder(stream);

    const microphone = this.audioContext.createMediaStreamSource(stream);

    const processorNode = this.audioContext.createScriptProcessor(4096, 1, 1);
    microphone.connect(processorNode);
    processorNode.connect(this.audioContext.destination);

    processorNode.onaudioprocess = (e) => {
      const data = e.inputBuffer.getChannelData(0);
      const audioSlice = Array.from(data);
      console.log(
        `Sending audio chunk: length=${audioSlice.length}, sample_rate=${this.audioContext.sampleRate}`
      );
      onAudio(data);
    };
  }

  public getRecorder() {
    return this.recorder;
  }
}
