// フェーズ状態を表す定数と型定義
// UI や API レスポンスで使用される phase 文字列を集中管理する

export const PHASE = {
  HEARING: 'hearing',
  SUMMARIZING: 'summarizing',
  CODE_GENERATION: 'code_generation',
  CODE_VALIDATION: 'code_validation',
  COMPLETED: 'completed',
} as const

export type Phase = typeof PHASE[keyof typeof PHASE]

export const isCompletedPhase = (p?: string | null): p is typeof PHASE.COMPLETED => p === PHASE.COMPLETED
