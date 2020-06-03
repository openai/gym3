#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <stdint.h>
#include <math.h>
#include <assert.h>
#include "libenv.h"


void fatal(const char *fmt, ...) {
    printf("fatal: ");
    va_list args;
    va_start(args, fmt);
    vprintf(fmt, args);
    va_end(args);
    exit(EXIT_FAILURE);
}

#define fassert(cond)                                                          \
    do {                                                                       \
        if (!(cond)) {                                                         \
            printf("fassert failed %s at %s:%d\n", #cond, __FILE__, __LINE__); \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

struct environment {
    int num;
    int step_count;
    int episode_len;
    struct libenv_tensortype ob_type;
    struct libenv_tensortype ac_type;
    struct libenv_buffers *bufs;
    uint8_t *sequence;
};

int libenv_version() {
    return LIBENV_VERSION;
}

libenv_env *libenv_make(int num, const struct libenv_options options) {
    int n_actions = 10;
    int episode_len = 100;
    uint8_t *sequence = NULL;
    int seq_count;

    for (int i = 0; i < options.count; i++) {
        struct libenv_option opt = options.items[i];
        if (strcmp(opt.name, "n_actions") == 0) {
            fassert(opt.dtype == LIBENV_DTYPE_INT32);
            n_actions = *(int32_t *)(opt.data);
        } else if (strcmp(opt.name, "episode_len") == 0) {
            fassert(opt.dtype == LIBENV_DTYPE_INT32);
            episode_len = *(int32_t *)(opt.data);
        } else if (strcmp(opt.name, "sequence") == 0) {
            fassert(opt.dtype == LIBENV_DTYPE_UINT8);
            seq_count = opt.count;
            sequence = (uint8_t *)(opt.data);
        } else {
            fatal("unrecognized option %s\n", opt.name);
        }
    }

    fassert(sequence != NULL);
    fassert(seq_count == episode_len);

    struct environment *e = calloc(1, sizeof(struct environment));
    e->num = num;
    e->step_count = 0;
    e->ob_type = (struct libenv_tensortype){
        .name = "ignore",
        .shape = {},
        .ndim = 0,
        .scalar_type = LIBENV_SCALAR_TYPE_REAL,
        .dtype = LIBENV_DTYPE_FLOAT32,
        .low.float32 = 0,
        .high.float32 = 0,
    };
    e->ac_type = (struct libenv_tensortype){
        .name = "action",
        .shape = {},
        .ndim = 0,
        .scalar_type = LIBENV_SCALAR_TYPE_DISCRETE,
        .dtype = LIBENV_DTYPE_UINT8,
        .low.uint8 = 0,
        .high.uint8 = n_actions,
    };
    e->episode_len = episode_len;
    e->sequence = sequence;
    return e;
}

int libenv_get_tensortypes(libenv_env *env, enum libenv_space_name name, struct libenv_tensortype *out_types) {
    struct environment *e = env;
    int count = 0;
    const struct libenv_tensortype *types = NULL;

    if (name == LIBENV_SPACE_OBSERVATION) {
        count = 1;
        types = &e->ob_type;
    } else if (name == LIBENV_SPACE_ACTION) {
        count = 1;
        types = &e->ac_type;
    }

    if (out_types != NULL && types != NULL) {
        for (int i = 0; i < count; i++) {
            out_types[i] = types[i];
        }
    }
    return count;
}

void libenv_set_buffers(libenv_env *env, struct libenv_buffers *bufs) {
    struct environment *e = env;
    e->bufs = bufs;
}

void libenv_observe(libenv_env *env) {
    struct environment *e = env;
    for (int env_idx = 0; env_idx < e->num; env_idx++) {
        float *ob = e->bufs->ob[env_idx];
        *ob = 0;
        uint8_t *ac = e->bufs->ac[env_idx];
        uint8_t correct_ac = e->sequence[e->step_count % e->episode_len];
        e->bufs->rew[env_idx] = (*ac == correct_ac);
        bool first = false;
        if (e->step_count % e->episode_len == 0) {
            first = true;
        }
        e->bufs->first[env_idx] = first;
    }
}

void libenv_act(libenv_env *env) {
    struct environment *e = env;
    e->step_count++;
}

void libenv_close(libenv_env *env) {
    struct environment *e = env;
    free(e);
}