import type {
  HealthResponse,
  AttacksResponse,
  RoundResult,
  PipelineResult,
  CompareResult,
  CompareModelsResult,
  AttackType,
} from "./types";
import {
  mockHealth,
  buildMockAttacks,
  buildMockRoundResult,
  buildMockPipeline,
  buildMockCompare,
  buildMockCompareModels,
} from "./mock";

export type {
  HealthResponse,
  AttacksResponse,
  RoundResult,
  PipelineResult,
  CompareResult,
  CompareModelsResult,
  AttackType,
};

export interface ApiResponse<T> {
  data: T;
  fromMock: boolean;
}

export async function fetchHealth(): Promise<ApiResponse<HealthResponse>> {
  try {
    const response = await fetch("/api/health");
    if (!response.ok) throw new Error("Failed to fetch health");
    const data: HealthResponse = await response.json();
    return { data, fromMock: false };
  } catch {
    return { data: mockHealth, fromMock: true };
  }
}

export async function fetchAttacks(): Promise<ApiResponse<AttacksResponse>> {
  try {
    const response = await fetch("/api/attacks");
    if (!response.ok) throw new Error("Failed to fetch attacks");
    const data: AttacksResponse = await response.json();
    return { data, fromMock: false };
  } catch {
    return { data: buildMockAttacks(), fromMock: true };
  }
}

export async function runRound(): Promise<ApiResponse<RoundResult>> {
  try {
    const response = await fetch("/api/run-round", { method: "POST" });
    if (!response.ok) throw new Error("Failed to run round");
    const data: RoundResult = await response.json();
    return { data, fromMock: false };
  } catch {
    return { data: buildMockRoundResult(1), fromMock: true };
  }
}

export async function runPipeline(): Promise<ApiResponse<PipelineResult>> {
  try {
    const response = await fetch("/api/run-pipeline", { method: "POST" });
    if (!response.ok) throw new Error("Failed to run pipeline");
    const data: PipelineResult = await response.json();
    return { data, fromMock: false };
  } catch {
    return { data: buildMockPipeline(), fromMock: true };
  }
}

export async function compareResponses(
  prompt: string
): Promise<ApiResponse<CompareResult>> {
  try {
    const response = await fetch("/api/compare", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ prompt }),
    });
    if (!response.ok) throw new Error("Failed to compare");
    const data: CompareResult = await response.json();
    return { data, fromMock: false };
  } catch {
    return { data: buildMockCompare(prompt), fromMock: true };
  }
}

export async function compareModels(): Promise<
  ApiResponse<CompareModelsResult>
> {
  try {
    const response = await fetch("/api/compare-models", { method: "POST" });
    if (!response.ok) throw new Error("Failed to compare models");
    const data: CompareModelsResult = await response.json();
    return { data, fromMock: false };
  } catch {
    return { data: buildMockCompareModels(), fromMock: true };
  }
}
