"use client";

export class AudioPlayer {
  private queue: Float32Array[] = [];
  private audioContext = new AudioContext();
  private playing = false;
  private output = this.audioContext.createMediaStreamDestination();

  public queueFragment(fragment: Float32Array) {
    this.queue.push(fragment);
    if (!this.playing) {
      this.processAudioPlaybackQueue().catch(console.error);
    }
  }

  public isPlaying() {
    return this.playing;
  }

  public getStream() {
    return this.output.stream;
  }

  public debug() {
    return {
      queue: this.queue,
      audioContext: this.audioContext,
      playing: this.playing,
      output: this.output,
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

  private async playAudioChunk(audioData: Float32Array, sampleRate: number) {
    return new Promise(async (resolve, reject) => {
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
          audioData.length,
          sampleRate
        );

        // Fill buffer with data
        audioBuffer.copyToChannel(audioData, 0);

        // Create source
        const source = this.audioContext.createBufferSource();
        source.buffer = audioBuffer;

        // Connect to destination and play
        source.connect(this.output);
        source.start(0);
        source.onended = resolve;
      } catch (err) {
        console.error("Audio playback error:", err);
      }
    });
  }
}
