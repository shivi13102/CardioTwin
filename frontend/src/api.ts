// src/api.ts — Axios API client for CardioTwin backend

import axios from 'axios';
import type { PatientInput, PredictResult, SimulateResult, SimulationDeltas } from './types';

const BASE_URL = 'http://localhost:8000';

const client = axios.create({
  baseURL: BASE_URL,
  headers: { 'Content-Type': 'application/json' },
});

export async function getSamplePatient(riskLevel: string): Promise<PatientInput> {
  const { data } = await client.get(`/sample-patient/${riskLevel}`);
  return data as PatientInput;
}

export async function predictTwin(patient: PatientInput): Promise<PredictResult> {
  const { data } = await client.post('/predict', patient);
  return data as PredictResult;
}

export async function simulateTwin(
  patient: PatientInput,
  deltas: SimulationDeltas,
  nSteps: number = 5
): Promise<SimulateResult> {
  const { data } = await client.post('/simulate', { patient, deltas, n_steps: nSteps });
  return data as SimulateResult;
}

export async function checkHealth(): Promise<{ status: string }> {
  const { data } = await client.get('/health');
  return data;
}
