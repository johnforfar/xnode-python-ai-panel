"use client";

export interface QueuedAudio {
  playAt: number;
  data: number[];
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

  public queueFragment(playAt: number, fragment: number[]) {
    console.log(
      `Received fragment to play at ${new Date(
        playAt * 1000
      ).toTimeString()} (${playAt * 1000 - Date.now()}ms from now)`
    );
    const nextUp = this.queue.at(-1);
    if (nextUp) {
      // Combine fragments to reduce overhead
      nextUp.data.concat(fragment);
    } else {
      this.queue.push({ playAt, data: fragment });
    }

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
      await this.playAudioChunk(nextChunk);
    } catch (err) {
      console.error("Error playing audio chunk:", err);
    }

    // Continue with next chunk
    this.processAudioPlaybackQueue();
  }

  private async playAudioChunk(audioData: QueuedAudio) {
    const delay = audioData.playAt * 1000 - Date.now();

    if (delay > 0) {
      await new Promise((resolve) => {
        console.log(`Playing next fragment in ${delay}ms`);
        setTimeout(resolve, delay);
      });
    } else {
      console.log(`Playing fragment ${delay}ms late`);
    }

    await new Promise(async (resolve, reject) => {
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

        const data = new Float32Array(audioData.data);

        // Create buffer
        const audioBuffer = this.audioContext.createBuffer(
          1, // mono
          data.length,
          24_000
        );

        // Fill buffer with data
        audioBuffer.copyToChannel(data, 0);

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
    });
  }
}
