/* The arithmetic of the policy while it plays, without a tensor library in between.
 *
 * An actor asks the network for one frame at a time: a handful of tree nodes
 * of 32 numbers, a recurrent step, now and then the map's two convolutions,
 * and at a decision a few small heads. Each of those is a few hundred
 * multiplications; through a tensor library each costs microseconds of
 * dispatch. This file holds the same operations as plain loops and BLAS calls.
 *
 * It computes what heroes/model/policy.py computes with torch bit for bit, and
 * every line here is written for that: a game played through this file is the
 * game torch would have played, with the same actions, probabilities and
 * values in every digit. What that takes:
 *  - products of matrices go through the same Accelerate routines torch calls,
 *    with the operands laid out as torch lays them out;
 *  - tanh and exp are torch's own (the SLEEF functions inside libtorch_cpu,
 *    handed over by fn_math), four numbers at a time as torch applies them;
 *  - everything between them is written in torch's order of operations, one
 *    rounding per operation: compile with -ffp-contract=off, so that no
 *    product is fused into a sum.
 * tests/test_fast.py holds every operation to exact equality with the model.
 *
 * Encodings of trees live in an arena of 32-float slots and are named by their
 * slot number; slot 0 is all zeros. The caller (heroes/model/fast.py) keeps
 * the memo from trees to slots and resets the arena when it drops the memo.
 * Recurrent states, queries, logits and map cells are the caller's own arrays.
 *
 * Built by heroes/model/fast.py with:
 *   cc -O3 -ffp-contract=off -DACCELERATE_NEW_LAPACK -shared -fPIC -framework Accelerate
 */
#include <Accelerate/Accelerate.h>
#include <arm_neon.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#define F 32   /* feature size */
#define S 64   /* recurrent state */

enum {
    P_BYTE, P_TEXT_WIH, P_TEXT_WHH, P_TEXT_BIH, P_TEXT_BHH, P_NUMBER_W, P_NUMBER_B, P_TREE_W, P_TREE_B,
    P_ORD_WIH, P_ORD_WHH, P_ORD_BIH, P_ORD_BHH, P_TERRAIN, P_ROAD, P_OBJECT_W, P_OBJECT_B,
    P_CONV1_W, P_CONV1_B, P_CONV2_W, P_CONV2_B, P_GLOBAL_W, P_GLOBAL_B,
    P_GRU_WIH, P_GRU_WHH, P_GRU_BIH, P_GRU_BHH, P_QUERY_W, P_QUERY_B, P_VALUE_W, P_VALUE_B,
    P_COORD0_W, P_COORD0_B, P_COORD2_W, P_COORD2_B, P_QUANT0_W, P_QUANT0_B, P_QUANT2_W, P_QUANT2_B, P_COUNT
};

typedef struct {
    float *p[P_COUNT];
    long size[P_COUNT];
    float *arena;
    long used, capacity;
    float *work;            /* scratch for the map, sequences and the heads */
    long work_size;
} Net;

/* torch's exp and tanh of four numbers (Sleef_expf4_u10 and Sleef_tanhf4_u10 of the libtorch_cpu in this process). */
typedef float32x4_t (*four)(float32x4_t);
static four exp4, tanh4;

int fn_math(void *exponential, void *tangent) {
    exp4 = (four)exponential;
    tanh4 = (four)tangent;
    return exp4 && tanh4 ? 0 : -1;
}

Net *fn_create(long capacity) {
    Net *net = calloc(1, sizeof(Net));
    if (!net) return NULL;
    net->capacity = capacity;
    net->arena = calloc((size_t)capacity * F, sizeof(float));
    net->used = 1;  /* slot 0 stays zero */
    if (!net->arena) { free(net); return NULL; }
    return net;
}

void fn_destroy(Net *net) {
    if (!net) return;
    for (int i = 0; i < P_COUNT; i++) free(net->p[i]);
    free(net->arena); free(net->work); free(net);
}

int fn_set(Net *net, int which, const float *data, long count) {
    if (which < 0 || which >= P_COUNT) return -1;
    free(net->p[which]);
    net->p[which] = malloc((size_t)count * sizeof(float));
    if (!net->p[which]) return -1;
    memcpy(net->p[which], data, (size_t)count * sizeof(float));
    net->size[which] = count;
    return 0;
}

int fn_ready(Net *net) {
    static const long expected[P_COUNT] = {
        257 * 8, 96 * 8, 96 * F, 96, 96, F * 2, F, F * 2 * F, F, 96 * F, 96 * F, 96, 96, 257 * 8, 257 * 4, 16 * 67, 16,
        16 * 33 * 9, 16, 16 * 16 * 9, 16, F * 256, F, 3 * S * 3 * F, 3 * S * S, 3 * S, 3 * S, F * S, F, S, 1,
        F * (S + F + 18), F, F, 1, F * (S + F + 4), F, 2 * F, 2 };
    for (int i = 0; i < P_COUNT; i++)
        if (!net->p[i] || net->size[i] != expected[i]) return -(i + 1);
    return exp4 && tanh4 ? 0 : -100;
}

long fn_reset(Net *net) { net->used = 1; return net->capacity; }
long fn_used(const Net *net) { return net->used; }
void fn_read(const Net *net, long index, float *out) { memcpy(out, net->arena + index * F, F * sizeof(float)); }

static inline float *slot(Net *net, long index) { return net->arena + index * F; }

static int room(Net *net, long floats) {
    if (net->work_size >= floats) return 0;
    free(net->work);
    net->work = malloc((size_t)floats * sizeof(float));
    net->work_size = net->work ? floats : 0;
    return net->work ? 0 : -1;
}

/* torch.tanh: its vector function over every four numbers, the last ones padded as torch pads them. */
static inline void tanh_all(float *values, long count) {
    long i = 0;
    for (; i + 4 <= count; i += 4) vst1q_f32(values + i, tanh4(vld1q_f32(values + i)));
    if (i < count) {
        float rest[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        memcpy(rest, values + i, (size_t)(count - i) * sizeof(float));
        vst1q_f32(rest, tanh4(vld1q_f32(rest)));
        memcpy(values + i, rest, (size_t)(count - i) * sizeof(float));
    }
}

/* torch.sigmoid over a multiple of four numbers: 1 / (1 + exp(0 - x)). */
static inline void sigmoid_all(float *values, long count) {
    const float32x4_t one = vdupq_n_f32(1.0f), zero = vdupq_n_f32(0.0f);
    for (long i = 0; i < count; i += 4)
        vst1q_f32(values + i, vdivq_f32(one, vaddq_f32(one, exp4(vsubq_f32(zero, vld1q_f32(values + i))))));
}

/* F.linear on one vector: out = weight (rows x columns) . x, then the bias added. */
static inline void affine(const float *weight, const float *bias, const float *x, long rows, long columns, float *out) {
    cblas_sgemv(CblasRowMajor, CblasNoTrans, (int)rows, (int)columns, 1.0f, weight, (int)columns, x, 1, 0.0f, out, 1);
    for (long i = 0; i < rows; i++) out[i] += bias[i];
}

/* F.linear on ``count`` vectors at once, as torch.addmm does it: out (count x rows) starts as the bias in every
 * row and gains x (count x columns) . weight^T in one product. */
static inline void affine_rows(const float *weight, const float *bias, const float *x, long count, long rows, long columns,
                               float *out) {
    for (long i = 0; i < count; i++) memcpy(out + i * rows, bias, (size_t)rows * sizeof(float));
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, (int)count, (int)rows, (int)columns, 1.0f, x, (int)columns,
                weight, (int)columns, 1.0f, out, (int)rows);
}

/* What torch's GRU cell makes of the two gate sums ``gi`` (from the input) and ``gh`` (from the state), gates
 * r, z, n in that order: r and z are sigmoids of gh + gi, n = tanh(gi + gh * r), out = (h - n) * z + n.
 * ``gh`` is overwritten; ``out`` may be ``h``. */
static void gru_gates(const float *gi, float *gh, const float *h, long hidden, float *out) {
    for (long i = 0; i < 2 * hidden; i++) gh[i] = gh[i] + gi[i];
    sigmoid_all(gh, 2 * hidden);
    float candidate[S];
    for (long i = 0; i < hidden; i++) candidate[i] = gi[2 * hidden + i] + gh[2 * hidden + i] * gh[i];
    tanh_all(candidate, hidden);
    for (long i = 0; i < hidden; i++) out[i] = (h[i] - candidate[i]) * gh[hidden + i] + candidate[i];
}

/* torch.gru_cell on one input and one state. */
static void gru(const float *wih, const float *whh, const float *bih, const float *bhh,
                const float *x, long inputs, const float *h, long hidden, float *out) {
    float gi[3 * S], gh[3 * S];
    affine_rows(wih, bih, x, 1, 3 * hidden, inputs, gi);
    affine_rows(whh, bhh, h, 1, 3 * hidden, hidden, gh);
    gru_gates(gi, gh, h, hidden, out);
}

/* torch.gru over a sequence of ``count`` inputs from a zero state, as nn.GRU runs it on the processor: the
 * input's gate sums for the whole sequence in one product, then the cell step by step. ``out`` gets the last state. */
static int gru_sequence(Net *net, const float *wih, const float *whh, const float *bih, const float *bhh,
                        const float *x, long count, long inputs, float *out) {
    if (room(net, count * 3 * F)) return -1;
    float *gi = net->work, gh[3 * F], h[F];
    affine_rows(wih, bih, x, count, 3 * F, inputs, gi);
    memset(h, 0, sizeof h);
    for (long i = 0; i < count; i++) {
        affine_rows(whh, bhh, h, 1, 3 * F, F, gh);
        gru_gates(gi + i * 3 * F, gh, h, F, h);
    }
    memcpy(out, h, sizeof h);
    return 0;
}

long fn_number(Net *net, float first, float second) {
    if (net->used >= net->capacity) return -1;
    float *out = slot(net, net->used), x[2] = {first, second};
    affine(net->p[P_NUMBER_W], net->p[P_NUMBER_B], x, F, 2, out);
    tanh_all(out, F);
    return net->used++;
}

/* A text from its UTF-8 bytes (the embedding table keeps row 0 for padding, so byte b reads row b + 1);
 * an empty text is slot 0. */
long fn_text(Net *net, const unsigned char *bytes, long count) {
    if (count == 0) return 0;
    if (net->used >= net->capacity) return -1;
    float small[64 * 8], *embedded = small;
    if (count > 64 && !(embedded = malloc((size_t)count * 8 * sizeof(float)))) return -1;
    for (long i = 0; i < count; i++) memcpy(embedded + i * 8, net->p[P_BYTE] + ((long)bytes[i] + 1) * 8, 8 * sizeof(float));
    int failed = gru_sequence(net, net->p[P_TEXT_WIH], net->p[P_TEXT_WHH], net->p[P_TEXT_BIH], net->p[P_TEXT_BHH],
                              embedded, count, 8, slot(net, net->used));
    if (embedded != small) free(embedded);
    return failed ? -1 : net->used++;
}

/* torch.sum over ``count`` rows of F numbers: row after row, and every sixteen rows the running sum is moved up a
 * level, the levels joined at the end (torch's cascade summation, which is the plain sum for fewer than sixteen). */
static void sum_rows(Net *net, const int *rows, long count, float *out) {
    float level[4][F];
    memset(level, 0, sizeof level);
    long i = 0;
    while (i + 16 <= count) {
        for (long j = 0; j < 16; j++, i++) {
            const float *row = slot(net, rows[i]);
            for (long k = 0; k < F; k++) level[0][k] = level[0][k] + row[k];
        }
        for (long j = 1; j < 4; j++) {
            for (long k = 0; k < F; k++) { level[j][k] = level[j][k] + level[j - 1][k]; level[j - 1][k] = 0.0f; }
            if ((i & (15L << (j * 4))) != 0) break;
        }
    }
    for (; i < count; i++) {
        const float *row = slot(net, rows[i]);
        for (long k = 0; k < F; k++) level[0][k] = level[0][k] + row[k];
    }
    for (long j = 1; j < 4; j++)
        for (long k = 0; k < F; k++) level[0][k] = level[0][k] + level[j][k];
    memcpy(out, level[0], F * sizeof(float));
}

/* A tree node: its children pooled (summed over the root of their number, or reduced in order), joined to its tag. */
long fn_node(Net *net, long tag, const int *children, long count, int ordered) {
    if (net->used >= net->capacity) return -1;
    float joined[2 * F];
    float *pooled = joined + F;
    memset(pooled, 0, F * sizeof(float));
    if (ordered && count) {
        float small[32 * F], *sequence = small;
        if (count > 32 && !(sequence = malloc((size_t)count * F * sizeof(float)))) return -1;
        for (long i = 0; i < count; i++) memcpy(sequence + i * F, slot(net, children[i]), F * sizeof(float));
        int failed = gru_sequence(net, net->p[P_ORD_WIH], net->p[P_ORD_WHH], net->p[P_ORD_BIH], net->p[P_ORD_BHH],
                                  sequence, count, F, pooled);
        if (sequence != small) free(sequence);
        if (failed) return -1;
    } else if (count) {
        sum_rows(net, children, count, pooled);
        const float scale = (float)sqrt((double)count);
        for (long k = 0; k < F; k++) pooled[k] = pooled[k] / scale;
    }
    memcpy(joined, slot(net, tag), F * sizeof(float));
    float *out = slot(net, net->used);
    affine(net->p[P_TREE_W], net->p[P_TREE_B], joined, F, 2 * F, out);
    tanh_all(out, F);
    return net->used++;
}

/* The recurrent step on a frame's three parts: its payload, the map and the seat's own view. */
void fn_step(Net *net, const float *state, float *out, long tree, long map, long own) {
    float x[3 * F];
    memcpy(x, slot(net, tree), F * sizeof(float));
    memcpy(x + F, slot(net, map), F * sizeof(float));
    memcpy(x + 2 * F, slot(net, own), F * sizeof(float));
    gru(net->p[P_GRU_WIH], net->p[P_GRU_WHH], net->p[P_GRU_BIH], net->p[P_GRU_BHH], x, 3 * F, state, S, out);
}

/* The recurrent step on a given input of three parts: for checks of the cell's arithmetic alone. */
void fn_cell(Net *net, const float *state, const float *x, float *out) {
    gru(net->p[P_GRU_WIH], net->p[P_GRU_WHH], net->p[P_GRU_BIH], net->p[P_GRU_BHH], x, 3 * F, state, S, out);
}

float fn_value(Net *net, const float *state) {
    float out;
    affine(net->p[P_VALUE_W], net->p[P_VALUE_B], state, 1, S, &out);
    return out;
}

void fn_query(Net *net, const float *state, float *query) {
    affine(net->p[P_QUERY_W], net->p[P_QUERY_B], state, F, S, query);
    tanh_all(query, F);
}

/* torch.stack(candidates).mv(query) / sqrt(F). Returns -1 when there is no memory for the stack. */
int fn_logits(Net *net, const float *query, const int *candidates, long count, float *out) {
    if (room(net, count * F)) return -1;
    for (long i = 0; i < count; i++) memcpy(net->work + i * F, slot(net, candidates[i]), F * sizeof(float));
    cblas_sgemv(CblasRowMajor, CblasNoTrans, (int)count, F, 1.0f, net->work, F, query, 1, 0.0f, out, 1);
    const float scale = (float)sqrt((double)F);
    for (long i = 0; i < count; i++) out[i] = out[i] / scale;
    return 0;
}

void fn_quantity(Net *net, const float *state, long features, const float *numeric, float *out) {
    float x[S + F + 4], hidden[F];
    memcpy(x, state, S * sizeof(float));
    memcpy(x + S, slot(net, features), F * sizeof(float));
    memcpy(x + S + F, numeric, 4 * sizeof(float));
    affine(net->p[P_QUANT0_W], net->p[P_QUANT0_B], x, F, S + F + 4, hidden);
    tanh_all(hidden, F);
    affine(net->p[P_QUANT2_W], net->p[P_QUANT2_B], hidden, 2, F, out);
}

/* A logit for every cell: the head reads the state, the target's features, the cell's map features and its
 * position, all cells in one product as torch's Linear takes them. */
int fn_coordinate(Net *net, const float *state, long target, const float *cells, const float *positions, long count, float *out) {
    const long inputs = S + F + 18;
    if (room(net, count * (inputs + F))) return -1;
    float *rows = net->work, *hidden = net->work + count * inputs;
    for (long i = 0; i < count; i++) {
        float *row = rows + i * inputs;
        memcpy(row, state, S * sizeof(float));
        memcpy(row + S, slot(net, target), F * sizeof(float));
        memcpy(row + S + F, cells + i * 16, 16 * sizeof(float));
        row[S + F + 16] = positions[i * 2];
        row[S + F + 17] = positions[i * 2 + 1];
    }
    affine_rows(net->p[P_COORD0_W], net->p[P_COORD0_B], rows, count, F, inputs, hidden);
    tanh_all(hidden, count * F);
    affine_rows(net->p[P_COORD2_W], net->p[P_COORD2_B], hidden, count, 1, F, out);
    return 0;
}

/* A 3x3 convolution with a border of zeros as torch computes it for one image: the input unfolded into
 * (channel, ky, kx) rows of one value per cell, and one product with the weights onto the bias.
 * ``input`` and ``out`` are channel after channel, each ``height`` rows of ``width``. */
static void convolve(const float *input, long channels, long width, long height, const float *weight, const float *bias,
                     float *unfolded, float *out) {
    const long cells = width * height, taps = 9 * channels;
    for (long c = 0; c < channels; c++)
        for (long ky = 0; ky < 3; ky++)
            for (long kx = 0; kx < 3; kx++) {
                float *row = unfolded + ((c * 3 + ky) * 3 + kx) * cells;
                for (long y = 0; y < height; y++) {
                    const long yy = y + ky - 1;
                    for (long x = 0; x < width; x++) {
                        const long xx = x + kx - 1;
                        row[y * width + x] = (yy < 0 || yy >= height || xx < 0 || xx >= width)
                            ? 0.0f : input[(c * height + yy) * width + xx];
                    }
                }
            }
    for (long o = 0; o < 16; o++)
        for (long i = 0; i < cells; i++) out[o * cells + i] = bias[o];
    /* torch: gemm in Fortran order, out (cells x 16) = unfolded (cells x taps) . weight (taps x 16) + out */
    cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, (int)cells, 16, (int)taps, 1.0f, unfolded, (int)cells,
                weight, (int)taps, 1.0f, out, (int)cells);
    tanh_all(out, cells * 16);
}

/* The map: per cell a terrain and a road number (each already raised by one, 0 for a cell not seen) and five
 * flags; the objects seen, each with its 67 attributes and its cell. Fills ``cells`` (cells x 16) and returns
 * the slot of the map's summary, or -1 when the arena or memory is exhausted. */
long fn_map(Net *net, const int *terrain, const int *road, const float *flags, const float *attributes,
            const int *object_cells, long objects, long width, long height, float *cells) {
    const long count = width * height;
    if (net->used >= net->capacity) return -1;
    /* work: the objects' encodings, the grid and both layers channel after channel, the unfolded input */
    if (room(net, objects * 16 + count * (33 + 16 + 16 + 9 * 33))) return -1;
    float *encoded = net->work, *grid = encoded + objects * 16, *first = grid + count * 33, *second = first + count * 16,
          *unfolded = second + count * 16;
    memset(grid + 12 * count, 0, (size_t)(16 * count) * sizeof(float));
    for (long i = 0; i < count; i++) {
        const float *ground = net->p[P_TERRAIN] + (long)terrain[i] * 8, *path = net->p[P_ROAD] + (long)road[i] * 4;
        for (long c = 0; c < 8; c++) grid[c * count + i] = ground[c];
        for (long c = 0; c < 4; c++) grid[(8 + c) * count + i] = path[c];
        for (long c = 0; c < 5; c++) grid[(28 + c) * count + i] = flags[i * 5 + c];
    }
    if (objects) {
        affine_rows(net->p[P_OBJECT_W], net->p[P_OBJECT_B], attributes, objects, 16, 67, encoded);
        tanh_all(encoded, objects * 16);
        for (long k = 0; k < objects; k++)
            for (long j = 0; j < 16; j++) grid[(12 + j) * count + object_cells[k]] += encoded[k * 16 + j];
    }
    convolve(grid, 33, width, height, net->p[P_CONV1_W], net->p[P_CONV1_B], unfolded, first);
    convolve(first, 16, width, height, net->p[P_CONV2_W], net->p[P_CONV2_B], unfolded, second);
    for (long i = 0; i < count; i++)
        for (long c = 0; c < 16; c++) cells[i * 16 + c] = second[c * count + i];
    /* the summary: every channel averaged over a 4 x 4 grid of regions, then one layer */
    float pooled[256];
    for (long c = 0; c < 16; c++)
        for (long i = 0; i < 4; i++) {
            const long y0 = (i * height) / 4, y1 = ((i + 1) * height + 3) / 4;
            for (long j = 0; j < 4; j++) {
                const long x0 = (j * width) / 4, x1 = ((j + 1) * width + 3) / 4;
                float sum = 0.0f;
                for (long y = y0; y < y1; y++)
                    for (long x = x0; x < x1; x++) sum += second[(c * height + y) * width + x];
                pooled[c * 16 + i * 4 + j] = sum / (float)(x1 - x0) / (float)(y1 - y0);
            }
        }
    float *out = slot(net, net->used);
    affine_rows(net->p[P_GLOBAL_W], net->p[P_GLOBAL_B], pooled, 1, F, 256, out);
    tanh_all(out, F);
    return net->used++;
}

/* -- the walks over semantic trees, for the interpreter that holds them ------------------------------------------
 *
 * A semantic tree is nested tuples: ("number", float), ("text", str), ("node", tag, children) and
 * ("ordered", tag, children); a tree built once is a policy._Kept, a tuple that remembers its hash and whether
 * equality is identity inside it. heroes/model/fast.py walks such trees for every frame: to check that a tree
 * is exact, and to find or compute the slot of every subtree through its memos. These are those two walks,
 * written against the interpreter's own objects; they visit what the Python walks visit, in their order, and
 * leave the memos as those leave them. Compiled only where Python.h is at hand (FASTNET_PYTHON).
 */
#ifdef FASTNET_PYTHON
#define PY_SSIZE_T_CLEAN
#include <Python.h>

static PyObject *kept_type, *name_exact;
static Py_ssize_t leaf_limit, node_limit, kept_limit;

static inline int named(PyObject *kind, const char *name) {
    return PyUnicode_CheckExact(kind) && PyUnicode_CompareWithASCIIString(kind, name) == 0;
}

/* policy._walk_exact: 1 when equality is identity for every subtree of ``tree``, 0 when not, -1 with an error set. */
static int walk_exact(PyObject *tree) {
    PyObject *few[256], **stack = few;
    Py_ssize_t size = 0, room = 256;
    int result = 1;
    stack[size++] = tree;
    while (size) {
        PyObject *item = stack[--size];
        if (!PyTuple_CheckExact(item)) {
            if ((PyObject *)Py_TYPE(item) == kept_type) {  /* checked when it was built */
                PyObject *flag = PyObject_GetAttr(item, name_exact);
                int truth = flag ? PyObject_IsTrue(flag) : -1;
                Py_XDECREF(flag);
                if (truth < 0) { result = -1; break; }
                if (truth) continue;
            }
            result = 0;
            break;
        }
        const Py_ssize_t length = PyTuple_GET_SIZE(item);
        PyObject *kind = length ? PyTuple_GET_ITEM(item, 0) : NULL;
        if (kind && named(kind, "number")) {
            PyObject *value = length == 2 ? PyTuple_GET_ITEM(item, 1) : NULL;
            if (!value || !PyFloat_CheckExact(value)) { result = 0; break; }
            const double number = PyFloat_AS_DOUBLE(value);
            if (number != number || (number == 0.0 && signbit(number))) { result = 0; break; }
        } else if (kind && named(kind, "text")) {
            if (length != 2 || !PyUnicode_CheckExact(PyTuple_GET_ITEM(item, 1))) { result = 0; break; }
        } else if (kind && (named(kind, "node") || named(kind, "ordered"))) {
            if (length != 3 || !PyUnicode_CheckExact(PyTuple_GET_ITEM(item, 1))
                    || !PyTuple_CheckExact(PyTuple_GET_ITEM(item, 2))) { result = 0; break; }
            PyObject *children = PyTuple_GET_ITEM(item, 2);
            const Py_ssize_t count = PyTuple_GET_SIZE(children);
            if (size + count > room) {
                while (size + count > room) room *= 2;
                PyObject **larger = malloc((size_t)room * sizeof(PyObject *));
                if (!larger) { PyErr_NoMemory(); result = -1; break; }
                memcpy(larger, stack, (size_t)size * sizeof(PyObject *));
                if (stack != few) free(stack);
                stack = larger;
            }
            for (Py_ssize_t i = 0; i < count; i++) stack[size++] = PyTuple_GET_ITEM(children, i);
        } else {
            result = 0;
            break;
        }
    }
    if (stack != few) free(stack);
    return result;
}

static PyObject *py_walk_exact(PyObject *self, PyObject *tree) {
    const int result = walk_exact(tree);
    if (result < 0) return NULL;
    return PyBool_FromLong(result);
}

/* memo[key] = slot, the memo emptied first once it holds ``limit`` entries (bounded memory; a miss only computes
 * again). 0, or -1 with an error set. */
static int remember(PyObject *memo, PyObject *key, long slot, Py_ssize_t limit) {
    if (PyDict_GET_SIZE(memo) >= limit) PyDict_Clear(memo);
    PyObject *value = PyLong_FromLong(slot);
    if (!value) return -1;
    const int failed = PyDict_SetItem(memo, key, value);
    Py_DECREF(value);
    return failed;
}

static long arena_full(void) {
    PyErr_SetString(PyExc_MemoryError, "The arena of tree encodings is full");
    return -1;
}

/* The slot of a text: its entry in ``texts``, or computed from its UTF-8 bytes and entered. -1 with an error set. */
static long text_slot(Net *net, PyObject *value, PyObject *texts) {
    PyObject *known = PyDict_GetItemWithError(texts, value);
    if (known) return PyLong_AsLong(known);
    if (PyErr_Occurred()) return -1;
    Py_ssize_t count;
    const char *bytes = PyUnicode_AsUTF8AndSize(value, &count);
    if (!bytes) return -1;
    const long slot = fn_text(net, (const unsigned char *)bytes, count);
    if (slot < 0) return arena_full();
    return remember(texts, value, slot, leaf_limit) ? -1 : slot;
}

/* FastPolicy._encode_exact: the slot of an exact tree's encoding, through the memos. -1 with an error set. */
static long encode_exact(Net *net, PyObject *tree, PyObject *texts, PyObject *numbers, PyObject *nodes, PyObject *kept) {
    const int is_kept = (PyObject *)Py_TYPE(tree) == kept_type;
    PyObject *identity = NULL;
    long slot;
    if (is_kept) {
        identity = PyLong_FromVoidPtr(tree);  /* what id() gives */
        if (!identity) return -1;
        PyObject *known = PyDict_GetItemWithError(kept, identity);
        if (known) {
            Py_DECREF(identity);
            return PyLong_AsLong(PyTuple_GET_ITEM(known, 1));
        }
        if (PyErr_Occurred()) { Py_DECREF(identity); return -1; }
    }
    PyObject *kind = PyTuple_GET_ITEM(tree, 0);
    if (named(kind, "text")) {
        slot = text_slot(net, PyTuple_GET_ITEM(tree, 1), texts);
    } else if (named(kind, "number")) {
        PyObject *value = PyTuple_GET_ITEM(tree, 1);
        PyObject *known = PyDict_GetItemWithError(numbers, value);
        if (known) {
            slot = PyLong_AsLong(known);
        } else if (PyErr_Occurred()) {
            slot = -1;
        } else {
            const double number = PyFloat_AS_DOUBLE(value);
            slot = fn_number(net, (float)copysign(log1p(fabs(number)), number), number == 0.0 ? 1.0f : 0.0f);
            if (slot < 0) slot = arena_full();
            else if (remember(numbers, value, slot, leaf_limit)) slot = -1;
        }
    } else {
        PyObject *known = PyDict_GetItemWithError(nodes, tree);
        if (known) {
            slot = PyLong_AsLong(known);
        } else if (PyErr_Occurred()) {
            slot = -1;
        } else {
            PyObject *children = PyTuple_GET_ITEM(tree, 2);
            const Py_ssize_t count = PyTuple_GET_SIZE(children);
            int few[64], *slots = few;
            if (count > 64 && !(slots = malloc((size_t)count * sizeof(int)))) {
                PyErr_NoMemory();
                Py_XDECREF(identity);
                return -1;
            }
            slot = 0;
            for (Py_ssize_t i = 0; i < count && slot >= 0; i++) {
                slot = encode_exact(net, PyTuple_GET_ITEM(children, i), texts, numbers, nodes, kept);
                slots[i] = (int)slot;
            }
            if (slot >= 0) {
                const long tag = text_slot(net, PyTuple_GET_ITEM(tree, 1), texts);
                if (tag < 0) {
                    slot = -1;
                } else {
                    slot = fn_node(net, tag, slots, count, named(kind, "ordered"));
                    if (slot < 0) slot = arena_full();
                    else if (remember(nodes, tree, slot, node_limit)) slot = -1;
                }
            }
            if (slots != few) free(slots);
        }
    }
    if (is_kept) {
        if (slot >= 0) {
            if (PyDict_GET_SIZE(kept) >= kept_limit) PyDict_Clear(kept);  /* a miss only looks the tree up again */
            PyObject *number = PyLong_FromLong(slot);
            PyObject *entry = number ? PyTuple_Pack(2, tree, number) : NULL;  /* holds the tree: its id names nothing else */
            Py_XDECREF(number);
            if (!entry || PyDict_SetItem(kept, identity, entry)) slot = -1;
            Py_XDECREF(entry);
        }
        Py_DECREF(identity);
    }
    return slot;
}

/* encode_root(net, tree, texts, numbers, nodes, kept): the slot of ``tree`` if it is exact, None if it is not
 * (the caller then encodes it without the memos). */
static PyObject *py_encode_root(PyObject *self, PyObject *const *arguments, Py_ssize_t count) {
    if (count != 6 || !PyTuple_Check(arguments[1]) || !PyDict_CheckExact(arguments[2]) || !PyDict_CheckExact(arguments[3])
            || !PyDict_CheckExact(arguments[4]) || !PyDict_CheckExact(arguments[5])) {
        PyErr_SetString(PyExc_TypeError, "encode_root(net, tree, texts, numbers, nodes, kept)");
        return NULL;
    }
    Net *net = PyLong_AsVoidPtr(arguments[0]);
    if (!net) {
        if (!PyErr_Occurred()) PyErr_SetString(PyExc_ValueError, "No network");
        return NULL;
    }
    PyObject *tree = arguments[1];
    int exact;
    if ((PyObject *)Py_TYPE(tree) == kept_type) {  /* policy._exact_tree: a kept tree was checked when it was built */
        PyObject *flag = PyObject_GetAttr(tree, name_exact);
        exact = flag ? PyObject_IsTrue(flag) : -1;
        Py_XDECREF(flag);
    } else {
        exact = walk_exact(tree);
    }
    if (exact < 0) return NULL;
    if (!exact) Py_RETURN_NONE;
    const long slot = encode_exact(net, tree, arguments[2], arguments[3], arguments[4], arguments[5]);
    if (slot < 0) {
        if (!PyErr_Occurred()) PyErr_SetString(PyExc_SystemError, "A tree could not be encoded");
        return NULL;
    }
    return PyLong_FromLong(slot);
}

/* The calls of every frame and every choice, as functions of this interpreter instead of through ctypes: the
 * same C functions with the same arguments (addresses and slot numbers as Python integers). */
static int addresses(PyObject *const *arguments, Py_ssize_t count, void **out) {
    for (Py_ssize_t i = 0; i < count; i++) {
        out[i] = PyLong_AsVoidPtr(arguments[i]);
        if (!out[i] && PyErr_Occurred()) return -1;
    }
    return 0;
}

static PyObject *py_used(PyObject *self, PyObject *argument) {
    Net *net = PyLong_AsVoidPtr(argument);
    if (!net) {
        if (!PyErr_Occurred()) PyErr_SetString(PyExc_ValueError, "No network");
        return NULL;
    }
    return PyLong_FromLong(fn_used(net));
}

/* step(net, state, out, tree, map, own) */
static PyObject *py_step(PyObject *self, PyObject *const *arguments, Py_ssize_t count) {
    void *at[3];
    if (count != 6) { PyErr_SetString(PyExc_TypeError, "step(net, state, out, tree, map, own)"); return NULL; }
    if (addresses(arguments, 3, at)) return NULL;
    const long tree = PyLong_AsLong(arguments[3]), map = PyLong_AsLong(arguments[4]), own = PyLong_AsLong(arguments[5]);
    if (PyErr_Occurred()) return NULL;
    fn_step(at[0], at[1], at[2], tree, map, own);
    Py_RETURN_NONE;
}

/* value(net, state) */
static PyObject *py_value(PyObject *self, PyObject *const *arguments, Py_ssize_t count) {
    void *at[2];
    if (count != 2) { PyErr_SetString(PyExc_TypeError, "value(net, state)"); return NULL; }
    if (addresses(arguments, 2, at)) return NULL;
    return PyFloat_FromDouble((double)fn_value(at[0], at[1]));
}

/* query(net, state, query) */
static PyObject *py_query(PyObject *self, PyObject *const *arguments, Py_ssize_t count) {
    void *at[3];
    if (count != 3) { PyErr_SetString(PyExc_TypeError, "query(net, state, query)"); return NULL; }
    if (addresses(arguments, 3, at)) return NULL;
    fn_query(at[0], at[1], at[2]);
    Py_RETURN_NONE;
}

/* logits(net, query, candidates, count, out) */
static PyObject *py_logits(PyObject *self, PyObject *const *arguments, Py_ssize_t count) {
    void *at[3], *out;
    if (count != 5) { PyErr_SetString(PyExc_TypeError, "logits(net, query, candidates, count, out)"); return NULL; }
    if (addresses(arguments, 3, at)) return NULL;
    const long number = PyLong_AsLong(arguments[3]);
    if (PyErr_Occurred()) return NULL;
    out = PyLong_AsVoidPtr(arguments[4]);
    if (!out && PyErr_Occurred()) return NULL;
    return PyLong_FromLong(fn_logits(at[0], at[1], at[2], number, out));
}

static PyMethodDef walks[] = {
    {"walk_exact", py_walk_exact, METH_O, "Whether equality is identity for every subtree of a semantic tree."},
    {"encode_root", (PyCFunction)(void (*)(void))py_encode_root, METH_FASTCALL,
     "The arena slot of an exact tree's encoding through the memos, or None for a tree that is not exact."},
    {"used", py_used, METH_O, "fn_used"},
    {"step", (PyCFunction)(void (*)(void))py_step, METH_FASTCALL, "fn_step"},
    {"value", (PyCFunction)(void (*)(void))py_value, METH_FASTCALL, "fn_value"},
    {"query", (PyCFunction)(void (*)(void))py_query, METH_FASTCALL, "fn_query"},
    {"logits", (PyCFunction)(void (*)(void))py_logits, METH_FASTCALL, "fn_logits"},
    {NULL, NULL, 0, NULL}};

/* Called once, with the interpreter lock held: policy._Kept and the memos' limits. Returns (walk_exact, encode_root,
 * used, step, value, query, logits). */
PyObject *fn_python(PyObject *kept, long leaves, long nodes, long kepts) {
    Py_XINCREF(kept);
    Py_XSETREF(kept_type, kept);
    if (!name_exact && !(name_exact = PyUnicode_InternFromString("exact"))) return NULL;
    leaf_limit = leaves; node_limit = nodes; kept_limit = kepts;
    PyObject *functions = PyTuple_New(7);
    if (!functions) return NULL;
    for (Py_ssize_t i = 0; i < 7; i++) {
        PyObject *function = PyCFunction_New(&walks[i], NULL);
        if (!function) { Py_DECREF(functions); return NULL; }
        PyTuple_SET_ITEM(functions, i, function);
    }
    return functions;
}
#endif
