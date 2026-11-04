/// <reference types="vite/client" />

/** The training worker (a separate file, or inline in the single-file build: see vite.config.ts). */
declare module '@trainer-worker' {
  const TrainerWorker: { new (options?: { name?: string }): Worker };
  export default TrainerWorker;
}

/** The mini-LLM training worker. */
declare module '@llm-worker' {
  const LLMWorker: { new (options?: { name?: string }): Worker };
  export default LLMWorker;
}
