import { describe, expect, it } from 'vitest';
import {
  extractUsage,
  isValidUsage,
  normalizeUsage,
  toOpenAIUsage,
} from './usage-normalize';

describe('isValidUsage', () => {
  it('识别 OpenAI 风格用量', () => {
    expect(isValidUsage({ prompt_tokens: 100 })).toBe(true);
    expect(isValidUsage({ completion_tokens: 50 })).toBe(true);
    expect(isValidUsage({ total_tokens: 150 })).toBe(true);
    expect(isValidUsage({ prompt_cache_hit_tokens: 80 })).toBe(true);
    expect(isValidUsage({ prompt_tokens_details: { cached_tokens: 60 } })).toBe(true);
  });

  it('识别 Anthropic 风格用量', () => {
    expect(isValidUsage({ input_tokens: 200 })).toBe(true);
    expect(isValidUsage({ output_tokens: 100 })).toBe(true);
  });

  it('识别 Google Vertex AI / Gemini 原生用量字段', () => {
    expect(isValidUsage({ promptTokenCount: 100 })).toBe(true);
    expect(isValidUsage({ candidatesTokenCount: 50 })).toBe(true);
    expect(isValidUsage({ totalTokenCount: 150 })).toBe(true);
    expect(isValidUsage({ cachedContentTokenCount: 30 })).toBe(true);
    expect(isValidUsage({ thoughtsTokenCount: 20 })).toBe(true);
  });

  it('识别内嵌 usageMetadata 的包装对象', () => {
    expect(isValidUsage({
      usageMetadata: {
        promptTokenCount: 300,
        candidatesTokenCount: 40,
      },
    })).toBe(true);
  });

  it('非对象或无 token 字段返回 false', () => {
    expect(isValidUsage(null)).toBe(false);
    expect(isValidUsage(undefined)).toBe(false);
    expect(isValidUsage('invalid')).toBe(false);
    expect(isValidUsage([])).toBe(false);
    expect(isValidUsage({})).toBe(false);
    expect(isValidUsage({ model: 'gemini-1.5-pro' })).toBe(false);
  });
});

describe('normalizeUsage', () => {
  it('读取 DeepSeek / OpenAI 兼容格式', () => {
    const usage = normalizeUsage({
      prompt_tokens: 1200,
      prompt_cache_hit_tokens: 900,
      prompt_cache_miss_tokens: 300,
      completion_tokens: 80,
      total_tokens: 1280,
    });
    expect(usage).toEqual({
      prompt_tokens: 1200,
      cache_hit_tokens: 900,
      cache_miss_tokens: 300,
      completion_tokens: 80,
      total_tokens: 1280,
      think_tokens: 0,
    });
  });

  it('读取带有 cached_tokens 的 OpenAI prompt_tokens_details', () => {
    const usage = normalizeUsage({
      prompt_tokens: 1000,
      completion_tokens: 20,
      total_tokens: 1020,
      prompt_tokens_details: { cached_tokens: 700 },
    });
    expect(usage).toMatchObject({
      prompt_tokens: 1000,
      cache_hit_tokens: 700,
      cache_miss_tokens: 300,
      total_tokens: 1020,
    });
  });

  it('处理 Anthropic 格式（input_tokens 不含缓存读写）', () => {
    const usage = normalizeUsage({
      input_tokens: 100,
      cache_creation_input_tokens: 200,
      cache_read_input_tokens: 700,
      output_tokens: 50,
    });
    expect(usage).toEqual({
      prompt_tokens: 1000,
      cache_hit_tokens: 700,
      cache_miss_tokens: 300,
      completion_tokens: 50,
      total_tokens: 1050,
      think_tokens: 0,
    });
  });

  it('处理 Google Vertex AI / Gemini 包装在 usageMetadata 里的格式', () => {
    const usage = normalizeUsage({
      usageMetadata: {
        promptTokenCount: 1000,
        candidatesTokenCount: 40,
        totalTokenCount: 1040,
        cachedContentTokenCount: 300,
      },
    });
    expect(usage).toEqual({
      prompt_tokens: 1000,
      cache_hit_tokens: 300,
      cache_miss_tokens: 700,
      completion_tokens: 40,
      total_tokens: 1040,
      think_tokens: 0,
    });
  });

  it('处理 Google Vertex AI / Gemini 扁平 usageMetadata 格式', () => {
    const usage = normalizeUsage({
      promptTokenCount: 500,
      candidatesTokenCount: 120,
      totalTokenCount: 620,
      cachedContentTokenCount: 200,
      thoughtsTokenCount: 80,
    });
    expect(usage).toEqual({
      prompt_tokens: 500,
      cache_hit_tokens: 200,
      cache_miss_tokens: 300,
      completion_tokens: 120,
      total_tokens: 620,
      think_tokens: 80,
    });
  });

  it('Google Gemini 思考 token 不超过 candidatesTokenCount', () => {
    const usage = normalizeUsage({
      promptTokenCount: 200,
      candidatesTokenCount: 50,
      totalTokenCount: 250,
      thoughtsTokenCount: 999,
    });
    expect(usage?.think_tokens).toBe(50);
  });

  it('空值、全 0 或无效结构返回 null', () => {
    expect(normalizeUsage(null)).toBeNull();
    expect(normalizeUsage(undefined)).toBeNull();
    expect(normalizeUsage({})).toBeNull();
    expect(normalizeUsage({ prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 })).toBeNull();
    expect(normalizeUsage({ promptTokenCount: 0, candidatesTokenCount: 0 })).toBeNull();
    expect(normalizeUsage({ usageMetadata: { promptTokenCount: 0, candidatesTokenCount: 0 } })).toBeNull();
    expect(normalizeUsage('invalid')).toBeNull();
  });
});

describe('toOpenAIUsage', () => {
  it('正确转换为标准 OpenAI 形状并附带原始 raw', () => {
    const normalized = normalizeUsage({
      promptTokenCount: 600,
      candidatesTokenCount: 90,
      cachedContentTokenCount: 150,
      thoughtsTokenCount: 40,
    })!;
    const raw = { promptTokenCount: 600 };
    const result = toOpenAIUsage(normalized, raw);
    expect(result.prompt_tokens).toBe(600);
    expect(result.prompt_cache_hit_tokens).toBe(150);
    expect(result.prompt_cache_miss_tokens).toBe(450);
    expect(result.completion_tokens).toBe(90);
    expect(result.total_tokens).toBe(690);
    expect(result.prompt_tokens_details).toEqual({ cached_tokens: 150 });
    expect(result.completion_tokens_details).toEqual({ reasoning_tokens: 40 });
    expect(result.__raw).toBe(raw);
    expect(result.__normalized).toBe(true);
  });
});

describe('extractUsage', () => {
  it('从包含 usageMetadata 的对象中提取并归一化', () => {
    const extra = {
      usageMetadata: {
        promptTokenCount: 800,
        candidatesTokenCount: 100,
      },
    };
    const extracted = extractUsage(extra);
    expect(extracted).not.toBeNull();
    expect(extracted?.prompt_tokens).toBe(800);
    expect(extracted?.completion_tokens).toBe(100);
  });

  it('从 response.usageMetadata 中提取', () => {
    const extra = {
      response: {
        usageMetadata: {
          promptTokenCount: 400,
          candidatesTokenCount: 50,
          cachedContentTokenCount: 100,
        },
      },
    };
    const extracted = extractUsage(extra);
    expect(extracted?.prompt_tokens).toBe(400);
    expect(extracted?.prompt_cache_hit_tokens).toBe(100);
    expect(extracted?.completion_tokens).toBe(50);
  });

  it('无有效用量时返回 null', () => {
    expect(extractUsage({})).toBeNull();
    expect(extractUsage({ extra: 'something' })).toBeNull();
  });
});
