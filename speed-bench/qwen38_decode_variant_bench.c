#include "ds4.h"

#include <errno.h>
#include <limits.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* Balanced same-engine Qwen decode A/B for dispatch-time environment knobs.
 *
 * One engine, two sessions synced to the same prefix.  Every step decodes the
 * same token in both sessions, once per variant, alternating both the variant
 * order and the variant-to-session pairing, and aborts unless every
 * full-vocabulary logit row is bit-identical.  Model load and prefill are
 * outside the timers, so a 512-step comparison costs seconds instead of the
 * minutes a process-per-variant CLI comparison needs, and both variants see
 * the same thermal and memory state.  Only knobs read at dispatch time can be
 * compared: settings cached when the graph is allocated look identical here.
 */

enum {
    VARIANT_COUNT = 2,
    MAX_ENV = 16,
    DEFAULT_PREFIX_TOKENS = 2048,
    DEFAULT_WARMUP = 16,
    DEFAULT_MEASURED = 512,
};

typedef struct {
    char *name;
    char *value;   /* NULL: unset for this variant */
} env_setting;

typedef struct {
    env_setting items[MAX_ENV];
    int n;
} env_list;

typedef struct {
    const char *model_path;
    const char *ple_path;
    const char *prompt_path;
    int prefix_tokens;
    int ctx;
    int warmup;
    int measured;
    uint32_t prefill_chunk;
    env_list variant[VARIANT_COUNT];   /* 0 = control, 1 = candidate */
} bench_config;

static void usage(FILE *fp, const char *argv0) {
    fprintf(fp,
            "usage: %s --candidate-env NAME[=VALUE] [options]\n"
            "\n"
            "  -m, --model PATH          GGUF path (default: ds4flash.gguf)\n"
            "  --ple PATH                optional PLE sidecar\n"
            "  --prompt-file PATH        token source (default: ds4.c)\n"
            "  --candidate-env NAME[=V]  set NAME (default value 1) for the candidate,\n"
            "                            unset it for the control; repeatable\n"
            "  --control-env NAME=V      set NAME=V for the control instead of unsetting\n"
            "  --prefix-tokens N         prefill length (default: 2048)\n"
            "  --prefill-chunk N         prefill chunk size (default: engine default)\n"
            "  --ctx N                   session allocation (default: prefix + steps + 1)\n"
            "  --warmup N                untimed steps per variant (default: 16)\n"
            "  --tokens N                measured steps per variant (default: 512)\n",
            argv0);
}

static const char *need_arg(int *i, int argc, char **argv, const char *opt) {
    if (*i + 1 >= argc) {
        fprintf(stderr, "qwen38-decode-variant-bench: %s requires an argument\n", opt);
        exit(2);
    }
    return argv[++*i];
}

static int parse_int_arg(const char *value, const char *opt, int minimum) {
    char *end = NULL;
    errno = 0;
    long parsed = strtol(value, &end, 10);
    if (errno != 0 || value[0] == '\0' || !end || *end != '\0' ||
        parsed < minimum || parsed > INT_MAX) {
        fprintf(stderr, "qwen38-decode-variant-bench: invalid value for %s: %s\n", opt, value);
        exit(2);
    }
    return (int)parsed;
}

static void env_list_add(env_list *list, const char *spec, bool require_value, const char *opt) {
    if (list->n >= MAX_ENV) {
        fprintf(stderr, "qwen38-decode-variant-bench: too many %s settings\n", opt);
        exit(2);
    }
    const char *eq = strchr(spec, '=');
    if (spec[0] == '\0' || spec[0] == '=' || (require_value && !eq)) {
        fprintf(stderr, "qwen38-decode-variant-bench: %s needs NAME%s: %s\n",
                opt, require_value ? "=VALUE" : "[=VALUE]", spec);
        exit(2);
    }
    env_setting *e = &list->items[list->n++];
    const size_t name_len = eq ? (size_t)(eq - spec) : strlen(spec);
    const char *value = eq ? eq + 1 : "1";
    e->name = malloc(name_len + 1u);
    e->value = malloc(strlen(value) + 1u);
    if (!e->name || !e->value) {
        fprintf(stderr, "qwen38-decode-variant-bench: out of memory\n");
        exit(1);
    }
    memcpy(e->name, spec, name_len);
    e->name[name_len] = '\0';
    strcpy(e->value, value);
}

static bench_config parse_options(int argc, char **argv) {
    bench_config cfg = {
        .model_path = "ds4flash.gguf",
        .prompt_path = "ds4.c",
        .prefix_tokens = DEFAULT_PREFIX_TOKENS,
        .warmup = DEFAULT_WARMUP,
        .measured = DEFAULT_MEASURED,
    };
    for (int i = 1; i < argc; i++) {
        const char *arg = argv[i];
        if (!strcmp(arg, "-h") || !strcmp(arg, "--help")) {
            usage(stdout, argv[0]);
            exit(0);
        } else if (!strcmp(arg, "-m") || !strcmp(arg, "--model")) {
            cfg.model_path = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--ple")) {
            cfg.ple_path = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--prompt-file")) {
            cfg.prompt_path = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--candidate-env")) {
            env_list_add(&cfg.variant[1], need_arg(&i, argc, argv, arg), false, arg);
        } else if (!strcmp(arg, "--control-env")) {
            env_list_add(&cfg.variant[0], need_arg(&i, argc, argv, arg), true, arg);
        } else if (!strcmp(arg, "--prefix-tokens")) {
            cfg.prefix_tokens = parse_int_arg(need_arg(&i, argc, argv, arg), arg, 1);
        } else if (!strcmp(arg, "--prefill-chunk")) {
            cfg.prefill_chunk = (uint32_t)parse_int_arg(need_arg(&i, argc, argv, arg), arg, 1);
        } else if (!strcmp(arg, "--ctx")) {
            cfg.ctx = parse_int_arg(need_arg(&i, argc, argv, arg), arg, 1);
        } else if (!strcmp(arg, "--warmup")) {
            cfg.warmup = parse_int_arg(need_arg(&i, argc, argv, arg), arg, 0);
        } else if (!strcmp(arg, "--tokens") || !strcmp(arg, "--measured")) {
            cfg.measured = parse_int_arg(need_arg(&i, argc, argv, arg), arg, 1);
        } else {
            fprintf(stderr, "qwen38-decode-variant-bench: unknown option: %s\n", arg);
            usage(stderr, argv[0]);
            exit(2);
        }
    }
    if (cfg.variant[1].n == 0) {
        fprintf(stderr, "qwen38-decode-variant-bench: --candidate-env is required\n");
        exit(2);
    }
    const int64_t needed = (int64_t)cfg.prefix_tokens + cfg.warmup + cfg.measured + 1;
    if (cfg.ctx == 0) cfg.ctx = (int)needed;
    if (needed > cfg.ctx) {
        fprintf(stderr,
                "qwen38-decode-variant-bench: --ctx must exceed prefix + warmup + measured (minimum %lld)\n",
                (long long)needed);
        exit(2);
    }
    return cfg;
}

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1.0e9;
}

static char *read_text(const char *path) {
    FILE *fp = fopen(path, "rb");
    if (!fp) {
        fprintf(stderr, "qwen38-decode-variant-bench: failed to open %s: %s\n", path, strerror(errno));
        return NULL;
    }
    if (fseek(fp, 0, SEEK_END) != 0) { fclose(fp); return NULL; }
    long len = ftell(fp);
    if (len < 0 || fseek(fp, 0, SEEK_SET) != 0) { fclose(fp); return NULL; }
    char *text = malloc((size_t)len + 1u);
    if (!text || fread(text, 1, (size_t)len, fp) != (size_t)len) {
        fprintf(stderr, "qwen38-decode-variant-bench: failed to read %s\n", path);
        free(text);
        fclose(fp);
        return NULL;
    }
    text[len] = '\0';
    fclose(fp);
    return text;
}

/* Apply one variant: its own settings are set, the other variant's names are
 * unset unless this variant also names them. */
static int select_variant(const bench_config *cfg, int variant) {
    const env_list *mine = &cfg->variant[variant];
    const env_list *other = &cfg->variant[variant ^ 1];
    for (int i = 0; i < other->n; i++) {
        bool covered = false;
        for (int j = 0; j < mine->n; j++) covered |= strcmp(mine->items[j].name, other->items[i].name) == 0;
        if (!covered && unsetenv(other->items[i].name) != 0) {
            fprintf(stderr, "qwen38-decode-variant-bench: unsetenv %s failed\n", other->items[i].name);
            return 1;
        }
    }
    for (int i = 0; i < mine->n; i++) {
        if (setenv(mine->items[i].name, mine->items[i].value, 1) != 0) {
            fprintf(stderr, "qwen38-decode-variant-bench: setenv %s failed\n", mine->items[i].name);
            return 1;
        }
    }
    return 0;
}

static uint32_t float_bits(float value) {
    uint32_t bits = 0;
    memcpy(&bits, &value, sizeof(bits));
    return bits;
}

static int compare_frontier(ds4_session *s0, ds4_session *s1, float *l0, float *l1,
                            int vocab, int eos, size_t step, int *token_out) {
    const size_t row_bytes = (size_t)vocab * sizeof(l0[0]);
    if (ds4_session_pos(s0) != ds4_session_pos(s1)) {
        fprintf(stderr, "qwen38-decode-variant-bench: position mismatch at step %zu\n", step);
        return 1;
    }
    memset(l0, 0xa5, row_bytes);
    memset(l1, 0x5a, row_bytes);
    if (ds4_session_copy_logits(s0, l0, vocab) != vocab ||
        ds4_session_copy_logits(s1, l1, vocab) != vocab) {
        fprintf(stderr, "qwen38-decode-variant-bench: failed to copy logits at step %zu\n", step);
        return 1;
    }
    if (memcmp(l0, l1, row_bytes) != 0) {
        size_t first = SIZE_MAX, differing = 0;
        for (int i = 0; i < vocab; i++) {
            if (memcmp(&l0[i], &l1[i], sizeof(float)) != 0) {
                if (first == SIZE_MAX) first = (size_t)i;
                differing++;
            }
        }
        fprintf(stderr,
                "qwen38-decode-variant-bench: raw logit mismatch at step %zu: differing=%zu/%d "
                "top0=%d top1=%d first id=%zu control=%a (0x%08x) candidate=%a (0x%08x)\n",
                step, differing, vocab, ds4_session_argmax(s0), ds4_session_argmax(s1),
                first, l0[first], (unsigned)float_bits(l0[first]),
                l1[first], (unsigned)float_bits(l1[first]));
        return 1;
    }
    const int t0 = ds4_session_argmax_excluding(s0, eos);
    const int t1 = ds4_session_argmax_excluding(s1, eos);
    if (t0 < 0 || t0 != t1) {
        fprintf(stderr, "qwen38-decode-variant-bench: token mismatch at step %zu: %d vs %d\n", step, t0, t1);
        return 1;
    }
    if (token_out) *token_out = t0;
    return 0;
}

static void print_env_list(FILE *fp, const env_list *list, const char *label) {
    fprintf(fp, "%s=", label);
    if (list->n == 0) fputs("(unset)", fp);
    for (int i = 0; i < list->n; i++)
        fprintf(fp, "%s%s=%s", i ? "," : "", list->items[i].name, list->items[i].value);
}

int main(int argc, char **argv) {
    const bench_config cfg = parse_options(argc, argv);
    char *text = read_text(cfg.prompt_path);
    if (!text) return 1;
    if (select_variant(&cfg, 0) != 0) { free(text); return 1; }

    ds4_engine_options opt = {
        .model_path = cfg.model_path,
        .ple_path = cfg.ple_path,
        .backend = DS4_BACKEND_METAL,
        .context_size = cfg.ctx,
        .prefill_chunk = cfg.prefill_chunk,
        .power_percent = 100,
        .warm_weights = true,
    };
    ds4_engine *engine = NULL;
    ds4_session *sessions[VARIANT_COUNT] = {0};
    ds4_tokens tokens = {0};
    float *logits[VARIANT_COUNT] = {0};
    char err[256] = {0};
    double elapsed[VARIANT_COUNT] = {0};
    size_t measured_tokens[VARIANT_COUNT] = {0};
    size_t exact_rows = 0;
    int rc = 1;

    if (ds4_engine_open(&engine, &opt) != 0) goto done;
    ds4_tokenize_text(engine, text, &tokens);
    free(text);
    text = NULL;
    if (tokens.len < cfg.prefix_tokens) {
        fprintf(stderr, "qwen38-decode-variant-bench: prompt has %d tokens; need %d\n",
                tokens.len, cfg.prefix_tokens);
        goto done;
    }
    const int vocab = ds4_engine_vocab_size(engine);
    ds4_tokens prefix = { .v = tokens.v, .len = cfg.prefix_tokens, .cap = cfg.prefix_tokens };
    for (int i = 0; i < VARIANT_COUNT; i++) {
        /* Both prefixes run under the control settings so the compared decode
         * steps start from identical state. */
        if (ds4_session_create(&sessions[i], engine, cfg.ctx) != 0 ||
            ds4_session_sync(sessions[i], &prefix, err, sizeof(err)) != 0) {
            fprintf(stderr, "qwen38-decode-variant-bench: session %d prefill failed: %s\n",
                    i, err[0] ? err : "unknown error");
            goto done;
        }
        logits[i] = malloc((size_t)vocab * sizeof(float));
        if (!logits[i]) goto done;
    }

    fprintf(stderr, "qwen38-decode-variant-bench: model=%s prefix=%d ctx=%d warmup=%d measured=%d ",
            cfg.model_path, cfg.prefix_tokens, cfg.ctx, cfg.warmup, cfg.measured);
    print_env_list(stderr, &cfg.variant[0], "control");
    fputc(' ', stderr);
    print_env_list(stderr, &cfg.variant[1], "candidate");
    fputc('\n', stderr);

    const int eos = ds4_token_eos(engine);
    const int total_steps = cfg.warmup + cfg.measured;
    for (int step = 0; step < total_steps; step++) {
        int token = -1;
        if (compare_frontier(sessions[0], sessions[1], logits[0], logits[1], vocab, eos,
                             (size_t)step, &token) != 0) goto done;
        exact_rows++;
        /* Even steps: control on session 0 then candidate on session 1.  Odd
         * steps reverse both the pairing and the order. */
        for (int order = 0; order < VARIANT_COUNT; order++) {
            const int session_i = order;
            const int variant_i = (order + step) & 1;
            if (select_variant(&cfg, variant_i) != 0) goto done;
            const double t0 = now_sec();
            if (ds4_session_eval(sessions[session_i], token, err, sizeof(err)) != 0) {
                fprintf(stderr, "qwen38-decode-variant-bench: decode failed at step=%d variant=%s: %s\n",
                        step, variant_i ? "candidate" : "control", err[0] ? err : "unknown error");
                goto done;
            }
            const int selected = ds4_session_argmax_excluding(sessions[session_i], eos);
            const double t1 = now_sec();
            if (selected < 0) goto done;
            if (step >= cfg.warmup) {
                elapsed[variant_i] += t1 - t0;
                measured_tokens[variant_i]++;
            }
        }
    }
    if (compare_frontier(sessions[0], sessions[1], logits[0], logits[1], vocab, eos,
                         (size_t)total_steps, NULL) != 0) goto done;
    exact_rows++;

    for (int v = 0; v < VARIANT_COUNT; v++) {
        printf("variant=%s tokens=%zu seconds=%.6f tokens_per_second=%.4f\n",
               v ? "candidate" : "control", measured_tokens[v], elapsed[v],
               elapsed[v] > 0.0 ? (double)measured_tokens[v] / elapsed[v] : 0.0);
    }
    if (elapsed[1] > 0.0 && measured_tokens[1] && measured_tokens[0]) {
        const double c = (double)measured_tokens[0] / elapsed[0];
        const double k = (double)measured_tokens[1] / elapsed[1];
        printf("speedup=%.4f percent=%+.3f exact_rows=%zu exact_floats=%zu\n",
               k / c, (k / c - 1.0) * 100.0, exact_rows, exact_rows * (size_t)vocab);
    }
    rc = 0;

done:
    for (int i = 0; i < VARIANT_COUNT; i++) {
        free(logits[i]);
        if (sessions[i]) ds4_session_free(sessions[i]);
    }
    ds4_tokens_free(&tokens);
    if (engine) ds4_engine_close(engine);
    free(text);
    return rc;
}
