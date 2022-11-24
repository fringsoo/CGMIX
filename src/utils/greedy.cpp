#include <iostream>
#include <cstdio>
#include <algorithm>
#include <vector>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <ctime>
using namespace std;

const int maxN = 20;
const int maxM = 20;
clock_t clockstart,clockend, cs, ce;
const int MAX_BATCH_SIZE = 35;
const int maxL = 305;

class GreedyActionSelector
{
    struct new_edge
    {
        int u, v;
        double f[maxM], maxf;
    } edges[maxN*maxN];

    double utils[maxN][maxM];        
    double messages[2][maxN*maxN][maxM];
    double msg[maxN][maxM];
    double joint0[maxM][maxM];
    double joint1[maxM][maxM];

    double maxsum_best_action[maxN];
    double maxsum_best_value;
            

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

    int cnt;

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
    
    void dfs_dp_graph(double *w_i, double *w_ij, int msg_iterations)
    {        
        maxsum_best_value = -1e10;
        memset(maxsum_best_action, 0, sizeof(maxsum_best_action));
        memset(edges, 0, sizeof(edges));
        memset(utils, 0, sizeof(utils));
        memset(messages, 0, sizeof(messages));
        memset(joint0, 0, sizeof(joint0));
        memset(joint1, 0, sizeof(joint1));

        int edge_count = 0;
        for (int u = 1; u < n; u++)
        {
            for (int v = u + 1; v <= n; v++)
            {
                edge_count += 1;
                edges[edge_count].u = u;
                edges[edge_count].v = v;
            }
        }
        
        for (int u = 1; u <= n; u++)
        {
            for (int i = 1; i <= m; i++)
            {
                //utils[u][i] = value_f(u,i); 
                utils[u][i] = value_f(u,i) * w_i[u-1];
            }
        }

        int ec;
        int iter;
        double joint0ei;
        double joint1ei;
        double message0_jsum;
        double message1_jsum;

        
        for (iter = 1; iter <= msg_iterations; iter++)
        {
            memset(msg, 0, sizeof(msg));
            for (ec=1; ec <= edge_count; ec++)
            {
                int u = edges[ec].u;
                int v = edges[ec].v;

                
                for (int i = 1; i <= m; i++)
                {
                    joint0ei = utils[u][i] - messages[1][ec][i];
                    joint1ei = utils[v][i] - messages[0][ec][i];
                    for (int j = 1; j <= m; j++)
                    {
                        //joint0[i][j] = joint0ei + value(u, v, i, j);
                        joint0[i][j] = joint0ei + value(u, v, i, j) * w_ij[(2*n-u)*(u-1)/2+v-u-1];
                        //joint1[i][j] = joint1ei + value(u, v, j, i);
                        joint1[i][j] = joint1ei + value(u, v, j, i) * w_ij[(2*n-u)*(u-1)/2+v-u-1];
                    }
                }

                message0_jsum = 0;
                message1_jsum = 0;
                for (int j = 1; j <= m; j++)
                {
                    messages[0][ec][j] = -1e10;
                    messages[1][ec][j] = -1e10;
                    
                    for (int i = 1; i<=m; i++)
                    {   
                        if (joint0[i][j] > messages[0][ec][j])
                            {
                                messages[0][ec][j] = joint0[i][j];
                            }
                        
                        if (joint1[i][j] > messages[1][ec][j])
                            {
                                messages[1][ec][j] = joint1[i][j];
                            }
                    }
                    message0_jsum += messages[0][ec][j];
                    message1_jsum += messages[1][ec][j];
                }

                for (int j = 1; j<=m; j++)
                {
                    messages[0][ec][j] -= message0_jsum / m;
                    messages[1][ec][j] -= message1_jsum / m;
                    msg[v][j] += messages[0][ec][j];
                    msg[u][j] += messages[1][ec][j];
                }
            }     
            
            double actions[n + 1];
            double tot_value = 0;
            
            for (int u = 1; u <= n; u++)
            {
                int action = -1;
                double util_u_max = -1e10;
                for (int i = 1; i <= m; i++)
                {
                    //utils[u][i] = value_f(u,i) + msg[u][i];
                    utils[u][i] = value_f(u,i) * w_i[u-1] + msg[u][i];
                    if (utils[u][i] > util_u_max)
                    {
                        util_u_max = utils[u][i];
                        action = i;
                    }
                }
                actions[u] = action;
                //tot_value += value_f(u, action);
                tot_value += value_f(u, action) * w_i[u-1];
            }

            for (ec=1; ec <= edge_count; ec++)
            {
                int u = edges[ec].u;
                int v = edges[ec].v;
                //tot_value += value(u, v, actions[u], actions[v]);
                tot_value += value(u, v, actions[u], actions[v]) * w_ij[(2*n-u)*(u-1)/2+v-u-1];
            }

            if (tot_value > maxsum_best_value)
            {
                maxsum_best_value = tot_value;
                //best_actions = actions;
                for (int u=1; u<=n;u++)
                {
                    maxsum_best_action[u - 1] = actions[u] - 1;
                    //best_actions[u - 1] = actions[u] - 1;
                }

            }
        
        }        
    }


    public:
    void solve_maxsum_graph(double *py_f, double *py_g, double *best_actions, int py_n, int py_m, int msg_iterations){
        n = py_n, m = py_m, f = py_f, g = py_g;
        
        double w_i[maxN], w_ij[maxN*maxN];
        memset(w_i, 0, sizeof(w_i));
        memset(w_ij, 0, sizeof(w_ij));
        for (int i=0; i<n; i++)
        {
            w_i[i] = 1;
        }
        for (int i=0; i < n * (n+1); i++)
        {
            w_ij[i] = 1;
        }

        //dfs_dp_graph(best_actions);
        dfs_dp_graph(w_i, w_ij, msg_iterations);
        //dfs_dp_graph();
        for (int u=0; u<n;u++)
        {
            best_actions[u] = maxsum_best_action[u];
        }
        //best_actions = maxsum_best_action;
    }

    
    
    int on_off_tag(int *use_relu)
    {
        int tag = 0;
        for (int l=0; l<len; l++)
        {
            tag += (1<<l) * use_relu[l];
        }
        return tag;
    }

    void tag_on_off(int tag, int *relu)
    {
        for (int l=0; l<len; l++)
        {
            relu[l] = (tag >> l) % 2;
        }
    }

    public:
    //void solve_graph(double *py_f, double *py_g, double *best_actions, double *py_w_1, double *py_w_final, double *py_bias, int py_n, int py_m, int py_l, double alpha, int msg_iterations, int onoff_configamount){
    void solve_graph(double *py_f, double *py_g, double *best_actions, double *wholeitert_timetotal, double *maxsum_timetotal, double *maxsum_iterounds, double *py_w_1, double *py_w_final, double *py_bias, int py_n, int py_m, int py_l, double alpha, int msg_iterations, int onoff_configamount, float epsilon_init, float epsilon_decay, int bav){
        n = py_n, m = py_m, len = py_l, f = py_f, g = py_g, w_1 = py_w_1, w_final = py_w_final, bias = py_bias;
        
        double best_value = -1e10;


        int explored[1 << len];
        memset(explored, 0, sizeof(explored));
        
        int use_relu[len];
        int real_use_relu[len];

        for (int l=0; l<len; l++)
        {
            use_relu[l] = 1;
        }

        int moving_tag = on_off_tag(use_relu);
        
        //Until local convergence
        int real_onoff_configamount = 0;
        if (onoff_configamount <= 0)
        {
            real_onoff_configamount = pow(2, len);
        }
        else
        {
            real_onoff_configamount = onoff_configamount;
        }
        for (int iter = 0; iter < real_onoff_configamount; iter++) //onoff rounds
        //for (int iter = 0; iter < onoff_configamount; iter++) //onoff rounds
        {   
            cs = clock();
            maxsum_iterounds[0] = iter + 1;
            
            int tag = on_off_tag(use_relu);
            explored[tag] = 1;
            
            double w_i[maxN];
            double w_ij[maxN*maxN];
            memset(w_i, 0, sizeof(w_i));
            memset(w_ij, 0, sizeof(w_ij));
            double wfinal_onoff;

            double res;
            res = 0;

            for (int l = 0; l < len; l++)
            {
                if (use_relu[l] > 0.5)
                    wfinal_onoff = w_final[l];
                else
                    wfinal_onoff = w_final[l] * alpha;
                res += bias[l] * wfinal_onoff;
                int cnt = 0;
                for (int i = 0; i < n; i++)
                {
                    w_i[i] += w_1[i * len + l] * wfinal_onoff;
                    for (int j = i + 1; j < n; j++)
                    {
                        w_ij[cnt] += w_1[(n + cnt) * len + l] * wfinal_onoff;
                        cnt += 1;
                    }
                }
            }

            clockstart = clock();
            dfs_dp_graph(w_i, w_ij, msg_iterations);
            clockend = clock();
            double maxsumtime=(double)(clockend-clockstart)/CLOCKS_PER_SEC;
            maxsum_timetotal[0] += maxsumtime;

            maxsum_best_value += res;
            if (bav == 0)
            {
                if (maxsum_best_value > best_value)
                {
                    for (int u=0; u<n;u++)
                    {
                        best_actions[u] = maxsum_best_action[u];
                    }
                    best_value = maxsum_best_value;
                }
            }
            // printf("%.4f,%.4f,%.4f\n",w_1[4],w_1[20],w_final[0]);
            // printf("%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,\n",w_i[0],w_i[1],w_i[2],w_i[3],w_ij[0],w_ij[1],w_ij[2],w_ij[3],w_ij[4],w_ij[5],w_ij[6],res);

            double check_maxsum_best_value = 0;
            for (int l = 0; l < len; l++){
                double node_value = 0;

                int cnt = 0;
                for (int u = 0; u < n; u++){
                    //node_value += value_f(u+1, maxsum_best_action[u+1]) * w_1[(n * n + n) / 2 * l + u];
                    node_value += value_f(u+1, maxsum_best_action[u]+1) * w_1[u * len + l];  
                    for (int v = u + 1; v < n; v++){
                        //node_value += value(u+1, v+1, maxsum_best_action[u+1], maxsum_best_action[v+1]) * w_1[(n * n + n) / 2 * l + n + cnt];
                        node_value += value(u+1, v+1, maxsum_best_action[u]+1, maxsum_best_action[v]+1) * w_1[(n + cnt) * len + l]; 
                        cnt += 1;
                    }
                }    

                node_value += bias[l];
                if (node_value > 0)
                {
                    real_use_relu[l] = 1;
                    check_maxsum_best_value += w_final[l] * node_value;
                }
                else
                {
                    real_use_relu[l] = 0;
                    check_maxsum_best_value += w_final[l] * node_value * alpha;
                }
            }

            
            //check_maxsum_best_value += res;
            if (bav==1)
            {
                if (check_maxsum_best_value > best_value)
                {
                    for (int u=0; u<n;u++)
                    {
                        best_actions[u] = maxsum_best_action[u];
                    }//
                    best_value = check_maxsum_best_value;
                }//
            }//
            
            // /////////////////////////////////////////////////////
            // for (int l=0; l<len;l++){
            //     printf("%d",use_relu[l]);
            // }//
            // printf("    %f\n",maxsum_best_value);
            // printf("maxsum_actions:");
            // for (int u = 0; u < n; u++){
            //     printf("%.0f,", maxsum_best_action[u]);
            // }//
            // printf("\n");
            // for (int l=0; l<len;l++){
            //     printf("%d",real_use_relu[l]);
            // }//
            // printf("    %f\n", check_maxsum_best_value);
            // printf("---\n");
            // /////////////////////////////////////////////////////

            if (on_off_tag(use_relu) == on_off_tag(real_use_relu))
            {
                if (onoff_configamount <= 0)
                    break;
            }
            //float epsilon = 0.4;
            float epsilon = epsilon_init * exp(- epsilon_decay * iter);
            float rand01 = rand() % 1000 / (float)(1000);
            bool real_as_new = (on_off_tag(use_relu)!=on_off_tag(real_use_relu)) && (! explored[on_off_tag(real_use_relu)]) && (rand01 < 1 - epsilon);
            //printf("%f\n", epsilon);
            if (real_as_new)
            {
                for (int l=0; l < len; l++)
                {
                    use_relu[l] = real_use_relu[l];
                }
            }
            else
            {   
                while (explored[moving_tag] && moving_tag > 0)
                {
                    moving_tag -= 1;
                }
                tag_on_off(moving_tag, use_relu);

            }

            ce = clock();
            double wholeitert=(double)(ce-cs)/CLOCKS_PER_SEC;
            wholeitert_timetotal[0] += wholeitert;
        }

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
        for (int k = 1; k < n; ++k) //k loop???
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
            is_on[i] = 0, new_on[i] = 1.;
        for (int l = 0; l < len; l++){
            cnt = 0;
            for (int i = 1; i <= n; i++){
                for (int k = 1; k <= m; k++){
                    // new_value_f[i][k] += value_f(i, k);
                    new_value_f[i][k] += value_f(i, k) * (new_on[l] - is_on[l]) * w_final[l] * w_1[(n * n + n) / 2 * l + i - 1];
                }
                for (int j = i + 1; j <= n; j++){
                    if (_id[i][j] > 0)
                        for (int k = 1; k <= m; k++)
                            for(int ll = 1; ll <= m; ll++){
                                // new_value[_id[i][j]][k][ll] = value(i, j, k, ll);
                                new_value[_id[i][j]][k][ll] = value(i, j, k, ll) * (new_on[l] - is_on[l]) * w_final[l] * w_1[(n * n + n) / 2 * l + n + cnt];
                            }
                    cnt ++;
                }
            }
            is_on[l] = new_on[l];
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
            // for (int i = 1; i <= n; i++)
            //     printf("action[%d] = %.1lf\n", i, best_actions[i - 1]);
            
            bool has_different = false;
            for (int l = 0; l < len; l++){
                double v = bias[l];
                for(int i = 1; i <= n; i++)
                    v += value_f(i, int(best_actions[i - 1]) + 1) * w_1[(n * n + n) / 2 * l + i - 1];
                cnt = 0;
                for(int i = 1; i <= n; i++)
                    for(int j = i + 1; j <= n; j++){
                        v += value(i, j, int(best_actions[i - 1]) + 1, int(best_actions[j - 1]) + 1) * w_1[(n * n + n) / 2 * l + n + cnt];
                        cnt ++;
                    }
                if (v < 0)
                    new_on[l] = alpha;
                else
                    new_on[l] = 1.0;
                if (fabs(is_on[l] - new_on[l]) > 1e-6){
                    cnt = 0;
                    for (int i = 1; i <= n; i++){
                        for (int k = 1; k <= m; k++){
                            new_value_f[i][k] += value_f(i, k) * (new_on[l] - is_on[l]) * w_final[l] * w_1[(n * n + n) / 2 * l + i - 1];
                        }
                        for (int j = i + 1; j <= n; j++){
                            if (_id[i][j] > 0)
                                for (int k = 1; k <= m; k++)
                                    for(int ll = 1; ll <= m; ll++){
                                        new_value[_id[i][j]][k][ll] = value(i, j, k, ll) * (new_on[l] - is_on[l]) * w_final[l] * w_1[(n * n + n) / 2 * l + n + cnt];
                                    }
                            cnt ++;
                        }
                    }
                    is_on[l] = new_on[l];
                    has_different = true;
                }
            }
            if (!has_different) break;
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

extern "C" void
maxsum_graph(double *py_f, double *py_g, double *best_actions, int bs, int n, int m, int msg_iterations)
{
    int t = (n + n * n) / 2;
#pragma omp parallel for schedule(dynamic, 1) num_threads(MAX_BATCH_SIZE)
    for (int i = 0; i < bs; i++)
        solver[i].solve_maxsum_graph(py_f + i * n * m, py_g + i * n * n * m * m, best_actions + i * n, n, m, msg_iterations);
}
extern "C" void
//greedy_graph(double *py_f, double *py_g, double *best_actions, double *w_1, double *w_final, double *bias, int bs, int n, int m, int l, double alpha, int msg_iterations, int onoff_configamount)
greedy_graph(double *py_f, double *py_g, double *best_actions, double *wholeitert_timetotal, double *maxsum_timetotal, double *maxsum_iterounds, double *w_1, double *w_final, double *bias, int bs, int n, int m, int l, double alpha, int msg_iterations, int onoff_configamount, float epsilon_init, float epsilon_decay, int bav)
{
    int t = (n + n * n) / 2;
#pragma omp parallel for schedule(dynamic, 1) num_threads(MAX_BATCH_SIZE)
    for (int i = 0; i < bs; i++)
        //solver[i].solve_graph(py_f + i * n * m, py_g + i * n * n * m * m, best_actions + i * n, w_1 + i * t * l, w_final + i * l, bias + i * l, n, m, l, alpha, msg_iterations, onoff_configamount);
        solver[i].solve_graph(py_f + i * n * m, py_g + i * n * n * m * m, best_actions + i * n, wholeitert_timetotal + i, maxsum_timetotal + i, maxsum_iterounds + i, w_1 + i * t * l, w_final + i * l, bias + i * l, n, m, l, alpha, msg_iterations, onoff_configamount, epsilon_init, epsilon_decay, bav);
}