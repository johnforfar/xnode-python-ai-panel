"use client";

export interface QueuedAudio {
  playAt: number;
  data: Float32Array;
}

export class AudioPlayer {
  private queue: QueuedAudio[] = [];
  private audioContext = new AudioContext();
  private playing = false;
  private output = this.audioContext.createMediaStreamDestination();
  private recorder = new MediaRecorder(this.output.stream);

  constructor() {
    this.recorder.start();
  }

  public queueFragment(playAt: number, fragment: Float32Array) {
    this.queue.push({ playAt, data: fragment });
    if (!this.playing) {
      this.processAudioPlaybackQueue().catch(console.error);
    }
  }

  public isPlaying() {
    return this.playing;
  }

  public getRecorder() {
    return this.recorder;
  }

  public debug() {
    return {
      queue: this.queue,
      audioContext: this.audioContext,
      playing: this.playing,
      output: this.output,
      recorder: this.recorder,
    };
  }

  private async processAudioPlaybackQueue() {
    const nextChunk = this.queue.shift();
    if (!nextChunk) {
      this.playing = false;
      return;
    }

    try {
      this.playing = true;
      await this.playAudioChunk(nextChunk, 24000);
    } catch (err) {
      console.error("Error playing audio chunk:", err);
    }

    // Continue with next chunk
    this.processAudioPlaybackQueue();
  }

  private async playAudioChunk(audioData: QueuedAudio, sampleRate: number) {
    return new Promise((resolve) =>
      setTimeout(resolve, Math.max(0, audioData.playAt * 1000 - Date.now()))
    ).then(
      () =>
        new Promise(async (resolve, reject) => {
          try {
            // Automatically resume audio context if suspended
            if (this.audioContext.state === "suspended") {
              try {
                console.log("Resuming audio context automatically");
                await this.audioContext.resume();
                console.log("Audio context resumed:", this.audioContext.state);
              } catch (err) {
                console.warn("Could not resume audio context:", err);
              }
            }

            // Create buffer
            const audioBuffer = this.audioContext.createBuffer(
              1, // mono
              audioData.data.length,
              sampleRate
            );

            // Fill buffer with data
            audioBuffer.copyToChannel(audioData.data, 0);

            // Create source
            const source = this.audioContext.createBufferSource();
            source.buffer = audioBuffer;

            // Connect to destination and play
            source.connect(this.output);
            source.connect(this.audioContext.destination);
            source.start(0);
            source.onended = resolve;
          } catch (err) {
            console.error("Audio playback error:", err);
          }
        })
    );
  }
}
