export interface HealthResponse {
  version: string;
}

export interface AttackType {
  category: string;
  description: string;
  sample_prompts: string[];
}

export interface AttacksResponse {
  attack_types: AttackType[];
}

export interface AttackResult {
  prompt: string;
  category: string;
  result: string;
  blocked: boolean;
  score: number;
}

export interface RoundResult {
  round: number;
  score: number;
  attack_results: AttackResult[];
  timestamp: number;
}

export interface PipelineResult {
  converged: boolean;
  total_rounds: number;
  final_score: number;
  rounds: RoundResult[];
}

export interface CompareResult {
  prompt: string;
  raw_response: string;
  hardened_response: string;
  raw_blocked: boolean;
  hardened_blocked: boolean;
}

export interface CompareModelsResult {
  model_a: string;
  model_b: string;
  rounds: Array<{
    round: number;
    model_a_score: number;
    model_b_score: number;
    model_a_blocked_count: number;
    model_b_blocked_count: number;
  }>;
}
