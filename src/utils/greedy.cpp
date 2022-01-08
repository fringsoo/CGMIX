#include <iostream>
#include <cstdio>
#include <algorithm>
#include <vector>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <ctime>
using namespace std;

const int maxN = 50;
const int maxM = 50;
const int MAX_BATCH_SIZE = 35;
const int maxL = 305;

class GreedyActionSelector
{
    struct edge
    {
        int u, v, tag;
        double f[maxM], maxf;
        edge *nxt;
    } pool[maxN * 3], *tp, *fst[maxN], *e[maxN];
    int n, m, len, seq[maxN][maxN][maxM];
    double *f, *g, *w_1, *w_final, *bias, delta[maxN][maxN], maxg2[maxN][maxN], maxg3[maxN][maxN][maxM];
    bool stale[maxN][maxN];


    vector<int> Tree_G[maxN];
    int _id[maxN][maxN];
    int vi[maxN];

    double is_on[maxL], new_on[maxL];
    double new_value_f[maxN][maxM];
    double new_value[maxN * 2][maxM][maxM];

    double dpv[maxN][maxM];

    void dfs_dp(int u)
    {
        vi[u] = 1;
        for (int j = 1; j <= m; j++)
            dpv[u][j] = new_value_f[u][j];
        for (int i = 0; i < Tree_G[u].size(); i++)
        {
            int v = Tree_G[u][i], e = _id[u][v];
            if (!vi[v])
            {
                dfs_dp(v);
                for (int j = 1; j <= m; j++)
                {
                    double max_value = -1e30;
                    for (int k = 1; k <= m; k++)
                        max_value = max(max_value, dpv[v][k] + new_value[e][j][k]);
                    dpv[u][j] += max_value;
                }
            }
        }
    }
    void dfs_construct(int u, int action, double *best_actions)
    {
        best_actions[u - 1] = action - 1;
        vi[u] = 1;
        for (int i = 0; i < Tree_G[u].size(); i++)
        {
            int v = Tree_G[u][i], e = _id[u][v];
            if (!vi[v])
            {
                double max_value = -1e30;
                int action_v = 0;
                for (int j = 1; j <= m; j++)
                    if (dpv[v][j] + new_value[e][action][j] > max_value)
                    {
                        max_value = dpv[v][j] + new_value[e][action][j];
                        action_v = j;
                    }
                dfs_construct(v, action_v, best_actions);
            }
        }
    }

    ////////////////////////////////////////////////////////////////////

    double value(int i, int j, int ai, int aj)
    {
        i -= 1, j -= 1, ai -= 1, aj -= 1;
        return g[i * n * m * m + j * m * m + ai * m + aj];
    }

    double value_f(int i,int ai)
    {
        i -= 1, ai -= 1;
        return f[i * m + ai];
    }

    void add_edge(int u, int v)
    {
        tp->u = u, tp->v = v;
        tp->nxt = fst[u], fst[u] = tp++;
    }

    void dp(edge *e, int tag)
    {
        if (e->tag == tag)
            return;
        e->tag = tag;
        e->maxf = -1e30;
        for (int i = 1; i <= m; ++i)
            e->f[i] = value_f(e->v, i);
        for (edge *son = fst[e->v]; son; son = son->nxt)
            if (son->v != e->u)
            {
                dp(son, tag);
                for (int i = 1; i <= m; ++i)
                {
                    double maxv = -1e30;
                    for (int j = 1; j <= m; ++j)
                        maxv = max(maxv, son->f[j] + value(e->v, son->v, i, j));
                    e->f[i] += maxv;
                }
            }
        for (int i = 1; i <= m; ++i)
            if (e->f[i] > e->maxf)
                e->maxf = e->f[i];
    }

    public:
    void solve(double *py_f, double *py_g, double *best_actions, double *py_w_1, double *py_w_final, double *py_bias, int py_n, int py_m, int py_l, double alpha){
        n = py_n, m = py_m, len = py_l, f = py_f, g = py_g, w_1 = py_w_1, w_final = py_w_final, bias = py_bias;
        int tmp = 0;
        memset(pool, 0, sizeof(pool)), tp = pool;
        memset(fst, 0, sizeof(fst));
        
        memset(_id, 0, sizeof(_id));
        for(int i = 0; i <= n; i++) Tree_G[i].clear();
        memset(vi, 0, sizeof(vi));
        for (int i = 1; i <= n; ++i)
            e[i] = tp, add_edge(0, i), dp(e[i], i);
        for (int i = 1; i <= n; ++i)
            for (int j = 1; j <= n; ++j)
            {
                maxg2[i][j] = -1e30;
                for (int ai = 1; ai <= m; ++ai)
                {
                    maxg3[i][j][ai] = -1e30;
                    for (int aj = 1; aj <=m; ++aj)
                        maxg3[i][j][ai] = max(maxg3[i][j][ai], value(i, j, ai, aj));
                    maxg2[i][j] = max(maxg2[i][j], maxg3[i][j][ai]);
                    for (int k = ai - 1; k >= 0; --k)
                        if (k == 0 || maxg3[i][j][seq[i][j][k]] > maxg3[i][j][ai])
                        {
                            seq[i][j][k + 1] = ai;
                            break;
                        }
                        else
                            seq[i][j][k + 1] = seq[i][j][k];
                }
                delta[i][j] = maxg2[i][j];
            }
        memset(stale, 0, sizeof(stale));
        for (int k = 1; k < n; ++k)
        {
            double max_delta = -1e30;
            int u, v;
            for (int i = 1; i <= n; ++i)
                for (int j = i + 1; j <= n; ++j)
                    if (e[i]->tag != e[j]->tag && !stale[i][j] && delta[i][j] > max_delta)
                        max_delta = delta[i][j], u = i, v = j;
            for (int i = 1; i <= n; ++i)
                for (int j = i + 1; j <= n; ++j)
                    if (e[i]->tag != e[j]->tag && stale[i][j] && maxg2[i][j] > max_delta)
                    {
                        stale[i][j] = false, delta[i][j] = -1e30;
                        for (int k = 1; k <= m; ++k)
                        {
                            int ai = seq[i][j][k];
                            if (maxg3[i][j][ai] + e[i]->f[ai] + e[j]->maxf > delta[i][j])
                                for (int aj = 1; aj <= m; ++aj)
                                    delta[i][j] = max(delta[i][j], value(i, j, ai, aj) + e[i]->f[ai] + e[j]->f[aj]);
                        }
                        delta[i][j] -= e[i]->maxf + e[j]->maxf;
                        if (delta[i][j] > max_delta)
                            max_delta = delta[i][j], u = i, v = j;
                    }
            add_edge(u, v), add_edge(v, u);
            Tree_G[u].push_back(v); Tree_G[v].push_back(u);
            tmp++;
            _id[u][v] = _id[v][u] = tmp;
            int tag_u = e[u]->tag, tag_v = e[v]->tag;
            for (int i = 1; i <= n; ++i)
                if (e[i]->tag == tag_u || e[i]->tag == tag_v)
                    dp(e[i], n + k);
            for (int i = 1; i <= n; ++i)
                for (int j = i + 1; j <= n; ++j)
                    if (e[i]->tag != e[j]->tag && (e[i]->tag == n + k || e[j]->tag == n + k))
                        stale[i][j] = true;
        }

        for (int i = 0; i < len; i++)
            is_on[i] = 1;
        double w_tot = 0;
        for (int i = 0; i < len; i++)
            w_tot += w_final[i];
        for (int i = 0; i < n; i++)
            for (int j = i + 1; j < n; j++) if (_id[i][j] > 0)
                for (int k = 0; k < m; k++)
                    for(int l = 0; l < m; l++){
                        new_value_f[i][k] = value_f(i + 1, k + 1);
                        new_value[_id[i][j]][k][l] = value(i + 1, j + 1, k + 1, l + 1);
                    }
        for (int iter = 0; iter < 50; iter++){ //// 50 times

            memset(vi, 0, sizeof(vi));
            dfs_dp(1);
            int action = 0;
            double max_value = -1e30;
            for (int j = 1; j <= m; j++)
                if (dpv[1][j] > max_value)
                {
                    max_value = dpv[1][j];
                    action = j;
                }
            memset(vi, 0, sizeof(vi));
            dfs_construct(1, action, best_actions);
            for (int l = 0; l < len; l++){
                double v = bias[l];
                for(int i = 1; i <= n; i++)
                    v += value_f(i, int(best_actions[i - 1]) + 1) / n;
                for(int i = 1; i <= n; i++)
                    for(int j = i + 1; j <= n; j++)
                        v += value(i, j, int(best_actions[i - 1]) + 1, int(best_actions[j - 1]) + 1) / ((n * n - n) / 2);
                if (v < 0)
                    new_on[l] = alpha;
                else
                    new_on[l] = 1.0;
                if (fabs(is_on[l] - new_on[l]) > 1e-6){
                    for (int i = 0; i < n; i++)
                        for (int j = i + 1; j < n; j++) if (_id[i][j] > 0)
                            for (int k = 0; k < m; k++)
                                for(int l = 0; l < m; l++){
                                    new_value_f[i][k] += value_f(i + 1, k + 1) * (new_on[l] - is_on[l]) * (w_final[l] / w_tot);
                                    new_value[_id[i][j]][k][l] = value(i + 1, j + 1, k + 1, l + 1) * (new_on[l] - is_on[l]) * (w_final[l] / w_tot);
                                }
                    is_on[l] = new_on[l];
                }
            }
        }
    }
};

GreedyActionSelector solver[MAX_BATCH_SIZE];

extern "C" void
greedy(double *py_f, double *py_g, double *best_actions, double *w_1, double *w_final, double *bias, int bs, int n, int m, int l, double alpha)
{
    int t = (n + n * n) / 2;
#pragma omp parallel for schedule(dynamic, 1) num_threads(MAX_BATCH_SIZE)
    for (int i = 0; i < bs; i++)
        solver[i].solve(py_f + i * n * m, py_g + i * n * n * m * m, best_actions + i * n, w_1 + i * t * l, w_final + i * l, bias + i * l, n, m, l, alpha);
}