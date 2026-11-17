/**
 * 通用用量归一化 —— 将各厂商（OpenAI、Anthropic、Google Vertex AI / Gemini 等）的用量数据统一映射为标准内部口径。
 *
 * 兼容来源：
 * - OpenAI / DeepSeek / 兼容中转：prompt_tokens / completion_tokens / prompt_cache_hit_tokens /
 *   prompt_cache_miss_tokens / prompt_tokens_details.cached_tokens / completion_tokens_details.reasoning_tokens
 * - OpenAI Responses：input_tokens / output_tokens / input_tokens_details.cached_tokens
 * - Anthropic Claude：input_tokens / output_tokens / cache_read_input_tokens / cache_creation_input_tokens
 * - Google Vertex AI / Gemini：usageMetadata.promptTokenCount / candidatesTokenCount /
 *   cachedContentTokenCount / thoughtsTokenCount / totalTokenCount
 */

export type NormalizedUsage = {
  /** 输入总量 = 命中 + 未命中 */
  prompt_tokens: number;
  cache_hit_tokens: number;
  cache_miss_tokens: number;
  completion_tokens: number;
  total_tokens: number;
  /** 思维链 token（provider 未提供时为 0，不做字符比例估算） */
  think_tokens: number;
};

/** 取第一个可用的有限数值；0 是有效值，undefined/null/NaN 才跳过 */
function firstNumber(...values: unknown[]): number | null {
  for (const value of values) {
    if (typeof value === 'number' && Number.isFinite(value)) return value;
    if (typeof value === 'string' && value.trim() !== '' && Number.isFinite(Number(value))) return Number(value);
  }
  return null;
}

function positive(value: number | null): number {
  return value != null && value > 0 ? value : 0;
}

function isObject(value: unknown): value is Record<string, any> {
  return !!value && typeof value === 'object' && !Array.isArray(value);
}

/**
 * 判断对象是否包含可识别的用量字段（放行 OpenAI、Anthropic 以及 Google Vertex AI 格式）
 */
export function isValidUsage(u: unknown): boolean {
  if (!isObject(u)) return false;
  if (isObject(u.usageMetadata) && isValidUsage(u.usageMetadata)) {
    return true;
  }
  return (
    typeof u.prompt_tokens === 'number' ||
    typeof u.completion_tokens === 'number' ||
    typeof u.total_tokens === 'number' ||
    typeof u.input_tokens === 'number' ||
    typeof u.output_tokens === 'number' ||
    typeof u.prompt_cache_hit_tokens === 'number' ||
    typeof u.cached_tokens === 'number' ||
    (u.prompt_tokens_details && typeof u.prompt_tokens_details.cached_tokens === 'number') ||
    typeof u.promptTokenCount === 'number' ||
    typeof u.candidatesTokenCount === 'number' ||
    typeof u.totalTokenCount === 'number' ||
    typeof u.cachedContentTokenCount === 'number' ||
    typeof u.thoughtsTokenCount === 'number'
  );
}

/**
 * 归一化任意 provider 的 usage 对象。
 * 全部 token 都为 0 或输入不是合法用量对象时返回 null（调用方应跳过该次记录，避免写入 0 token 污染条目）。
 */
export function normalizeUsage(raw: unknown): NormalizedUsage | null {
  if (!isObject(raw)) return null;
  let source: Record<string, any> = raw;
  // Gemini 会把 usage 包在 usageMetadata 里；同时兼容外层已含其它字段的形状
  if (isObject(source.usageMetadata)) {
    source = { ...source, ...source.usageMetadata };
  }

  const anthropic = firstNumber(source.cache_read_input_tokens, source.cache_creation_input_tokens) !== null;
  const gemini = firstNumber(
    source.promptTokenCount,
    source.candidatesTokenCount,
    source.totalTokenCount,
    source.cachedContentTokenCount,
    source.thoughtsTokenCount,
  ) !== null;

  const promptRaw = firstNumber(
    source.prompt_tokens,
    source.input_tokens,
    source.total_input_tokens,
    gemini ? source.promptTokenCount : undefined,
  );
  const completionRaw = firstNumber(
    source.completion_tokens,
    source.output_tokens,
    gemini ? source.candidatesTokenCount : undefined,
  );
  const hitRaw = firstNumber(
    source.prompt_cache_hit_tokens,
    source.cache_hit_tokens,
    source.prompt_tokens_details?.cached_tokens,
    source.input_tokens_details?.cached_tokens,
    source.cache_read_input_tokens,
    gemini ? source.cachedContentTokenCount : undefined,
    source.cached_tokens,
  );
  const missRaw = firstNumber(source.prompt_cache_miss_tokens, source.cache_miss_tokens);
  const totalRaw = firstNumber(source.total_tokens, source.totalTokenCount);
  const thinkRaw = firstNumber(
    source.completion_tokens_details?.reasoning_tokens,
    source.output_tokens_details?.reasoning_tokens,
    source.reasoning_tokens,
    source.thinking_tokens,
    source.thoughtsTokenCount,
  );

  const prompt = positive(promptRaw);
  const completion = positive(completionRaw);
  let hit = positive(hitRaw);
  let miss = missRaw != null && missRaw >= 0 ? missRaw : -1;

  if (miss < 0) {
    if (anthropic) {
      // Anthropic：input_tokens 只统计未缓存输入，缓存读/写单列
      const cacheCreate = positive(firstNumber(source.cache_creation_input_tokens));
      miss = prompt + cacheCreate;
    } else {
      miss = Math.max(0, prompt - hit);
    }
  }

  if (hit > prompt + miss && !anthropic && !gemini) {
    // 少数中转把 cached_tokens 重复计入 input_tokens，修正命中不超过输入
    hit = prompt + miss;
  }
  const promptTotal = Math.max(prompt, hit + miss);
  if (promptTotal > hit + miss) miss = promptTotal - hit;

  const computedTotal = hit + miss + completion;
  const total = Math.max(totalRaw != null && totalRaw > 0 ? totalRaw : 0, computedTotal);

  if (hit + miss + completion <= 0 && total <= 0) return null;

  const think = Math.min(positive(thinkRaw), completion > 0 ? completion : positive(thinkRaw));
  return {
    prompt_tokens: hit + miss,
    cache_hit_tokens: hit,
    cache_miss_tokens: miss,
    completion_tokens: completion,
    total_tokens: total,
    think_tokens: think,
  };
}

/**
 * 把归一化结果合成为扩展内部使用的 OpenAI 形状 usage 对象。
 * 原始 usage 挂在 __raw 上，供历史详情查看。
 */
export function toOpenAIUsage(normalized: NormalizedUsage, raw?: unknown): Record<string, any> {
  const usage: Record<string, any> = {
    prompt_tokens: normalized.prompt_tokens,
    completion_tokens: normalized.completion_tokens,
    total_tokens: normalized.total_tokens,
    prompt_cache_hit_tokens: normalized.cache_hit_tokens,
    prompt_cache_miss_tokens: normalized.cache_miss_tokens,
    __normalized: true,
  };
  if (normalized.cache_hit_tokens > 0) {
    usage.prompt_tokens_details = { cached_tokens: normalized.cache_hit_tokens };
  }
  if (normalized.think_tokens > 0) {
    usage.completion_tokens_details = { reasoning_tokens: normalized.think_tokens };
  }
  if (raw !== undefined) usage.__raw = raw;
  return usage;
}

/**
 * 从候选对象（如 extra、chunk 等）中按优先级提取首个有效 usage 并归一化合成 OpenAI 形状
 */
export function extractUsage(raw: unknown): Record<string, any> | null {
  if (!isObject(raw)) return null;
  const candidates = [
    raw.api_usage,
    raw.usage,
    raw.usageMetadata,
    raw.openai_usage,
    raw.token_usage,
    raw.data?.usage,
    raw.data?.usageMetadata,
    raw.response?.usage,
    raw.response?.usageMetadata,
  ];
  for (const c of candidates) {
    const normalized = normalizeUsage(c);
    if (normalized) return toOpenAIUsage(normalized, c);
  }
  const normalized = normalizeUsage(raw);
  if (normalized) return toOpenAIUsage(normalized, raw);
  return null;
}
