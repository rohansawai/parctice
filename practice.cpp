// Graph and DP patterns Practice

// DFS
#include<stdlib.h>
#include <vector>
#include <queue>
#include <set>
#include <unordered_set>
#include <numeric>

using namespace std;

void dfs(int u, vector<vector<int>>& adj, vector<bool>& vis){
    vis[u] = true;

    for(int v : adj[u]){
        if(!vis[v]){
            dfs(v, adj, vis); 
        }
    }
}

void dfs(int u, vector<vector<int>>& adj, vector<bool>& vis){
    vis[u] = true;

    for(auto v : adj[u]){
        if(!vis[v]){
            dfs(v, adj, vis);
        }
    }
}

void dfs(int u, vector<vector<int>>& adj, vector<bool>& vis){
    vis[u] = true;

    for(auto v : adj[u]){
        if(!vis[v]){
            dfs(v, adj, vis);
        }
    }
}


// Grid DFS

int dirs[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};

void gridDfs(int i, int j, vector<vector<char>>& grid){
    int n = grid.size(), m = grid[0].size();
    
    if(i<0 || j<0 || i>=n || j>=m || grid[i][j]=='0') return;

    for(auto itr : dirs){
        gridDfs(i+itr[0], j+itr[1], grid);
    }
}


int dir[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};

void gridDfs(int i, int j, vector<vector<char>> grid){
    int n = grid.size(), m = grid[0].size();

    if(i<0 || j<0 || i>=n || j>=m || grid[i][j]=='0') return;

    for(auto& d : dir){
        gridDfs(i+d[0], j+d[1], grid);
    }
}


int dir[4][2] = {{1,0}, {-1,0}, {0,1}, {0,-1}};

void gridDfs(int i, int j, vector<vector<char>>& grid){
    int n = grid.size(), m = grid[0].size();

    if(i<0 || j<0 || i>=n || j>=m || grid[i][j]=='0') return;

    grid[i][j] = '0';

    for(auto& d:dir){
        gridDfs(i+d[0], j+d[1], grid);
    }
}


int dir[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};

void gridDfs(int i, int j, vector<vector<char>>& grid){
    int n = grid.size(), m = grid[0].size();

    if(i<0 || j<0 || i>=n || j>=m || grid[i][j] == '0') return;

    grid[i][j] = '0';

    for(auto& d:dir){
        gridDfs(i+d[0], j+d[1], grid);
    }
}


int dir[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};

void gridDfs(int i, int j, vector<vector<char>>& grid){
    int n = grid.size(), m = grid[0].size();

    if(i<0 || j<0 || i>=n || j>=m || grid[i][j]=='0') return;

    grid[i][j] = '0';

    for(auto& d:dir){
        gridDfs(i+d[0], j+d[1], grid);
    }
}


// BFS

void bfs(int src, vector<vector<int>>& adj) {
    queue<int> q;
    vector<bool> vis(adj.size(), false);

    q.push(src);
    vis[src] = true;

    while(!q.empty()){
        int u = q.front(); q.pop();
        for(int v : adj[u]){
            if(!vis[v]){
                vis[v] = true;
                q.push(v);
            }
        }
    }
}


void bfs(int src, vector<vector<int>>& adj){
    queue<int> q;
    vector<bool> vis(adj.size(), false);

    q.push(src);
    vis[src] = true;

    while(!q.empty()){
        int u = q.front(); q.pop();
        for(int v : adj[u]){
            if(!vis[v]){
                vis[v] = true;
                q.push(v);
            }
        }
    }
}


void bfs(int src, vector<vector<int>>& adj){
    int n = adj.size();
    vector<bool> vis(n, false);

    queue<int> q; 
    q.push(src);
    vis[src] = true;

    while(!q.empty()){
        int x = q.front(); q.pop();

        for(auto& d:adj[x]){
            if(!vis[d]){
                q.push(d);
                vis[d] = true;
            }
        }
    }
}

void bfs(int src, vector<vector<int>> adj){
    int n = adj.size();
    vector<bool> vis(n, false);
    queue<int> q;

    q.push(src);
    vis[src] = true;

    while(!q.empty()){
        int u = q.front(); q.pop();

        for(auto& v:adj[u]){
            if(!vis[v]){
                vis[v] = true;
                q.push(v);
            }
        }
    }
}


// BFS Grid

int dirs[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};

void gridBfs(int i, int j, vector<vector<char>>& grid){
    int n = grid.size(), m = grid[0].size();
    
    queue<pair<int, int>> q;
    q.push({i, j});
    grid[i][j] = '0';

    while(!q.empty()){
        auto [x, y] = q.front(); q.pop();
        for(auto& d:dirs){
            int vi = x+d[0];
            int vj = y+d[1];
            if(vi>0 && vj >0 && vi<n && vj<m && grid[vi][vj]=='1'){
                grid[vi][vj] = '0';
                q.push({vi, vj});
            }
        }
    }
}




int dir[4][2] = {{1,0},{-1,0},{0,1},{0,-1}};

void gridBfs(int i, int j, vector<vector<char>>& grid){
    int n = grid.size(), m = grid[0].size();
    queue<pair<int, int>> q;
    q.push({i, j});
    grid[i][j] = '0';

    while(!q.empty()){
        auto [x,y] = q.front(); q.pop();

        for(auto& d:dir){
            int ui = x + d[0];
            int uj = y + d[1];
            if(ui>0 && uj>0 && ui<n && uj<m && grid[ui][uj] == '1'){
                grid[ui][uj] = '0';
                q.push({ui, uj});
            }
        }
    }
}



//  clone graphs DFS

unordered_map<Node*, Node*> mp;

Node* cloneGraph(Node* node){
    if(!node) return nullptr;
    if(mp.count(node)) return mp[node];

    Node* copy = new Node(node->val);
    mp[node] = copy;

    for(auto nei:node->neighbours){
        copy->neighbours.push_back(cloneGraph(nei));
    }
    return copy;
}



//  Dijkstra Algo

vector<int> djikstra(int n, vector<vector<pair<int, int>>>& adj, int src){
    vector<int> dist(n, INT_MAX);
    priority_queue<pair<int, int>, vector<pair<int, int>>, greater<>> pq;

    dist[src] = 0;
    pq.push({0, src});

    while(!pq.empty()){
        auto [d, u] = pq.top(); pq.pop();
        if(d>dist[u]) continue;

        for(auto [v, w]:adj[u]){
            if(dist[v] > d+w){
                pq.push({dist[v], v});
            }
        }
    }
    return dist;
}


// Djikstra Algo

vector<int> djikstra(int n, vector<vector<pair<int, int>>>& adj, int src){
    vector<int> dist(n, INT_MAX);
    priority_queue<pair<int, int>, vector<pair<int, int>>, greater<>> pq;

    dist[src] = 0;
    pq.push({0, src});

    while(!pq.empty()){
        auto [d, u] = pq.top(); pq.pop();
        if(d<dist[u]) continue;

        for(auto [v, w]:adj[u]){
            if(dist[v] > d+w){
                dist[v] = d+w;
                pq.push({dist[v], v});
            }
        }
    }
}


// Djikstra Algo using Set TC: E(logV)

vector<int> djikstra(int V, vector<vector<int>> adj[], int S){
    set<pair<int, int>> st;
    vector<int> dist(V, 1e9);

    st.insert({0, S});
    dist[S] = 0;

    while(!st.empty()){
        auto it = *(st.begin());
        int node = it.second;
        int dis = it.first;
        st.erase(it);

        for(auto it : adj[node]){
            int adjNode = it[0];
            int edgW = it[1];

            if(dis + edgW < dist[adjNode]){
                if(dist[adjNode] != 1e9){
                    st.erase({dist[adjNode], adjNode});
                }

                dist[adjNode] = dis + edgW;
                st.insert({dist[adjNode], adjNode});
            }
        }
    }
    return dist;
}



// No of ways to arrive at a destination

int countPaths(int n, vector<vector<int>>& roads){
    int mod = 1e9 + 7;

    vector<vector<pair<int, int>>> adj(n);
    for(auto& road : roads){
        adj[road[0]].push_back({road[1], road[2]});
        adj[road[1]].push_back({road[0], road[2]});
    }

    vector<int> counts(n, 0);
    vector<long long> dist(n, LLONG_MAX);

    priority_queue<pair<long long, int>, vector<pair<long long, int>>, greater<pair<long long, int>>> pq;
    pq.push({0, 0});
    counts[0]++;
    dist[0] = 0;

    while(!pq.empty()){
        auto [d, u] = pq.top(); pq.pop();
        if(d>dist[u])continue;

        for(auto [v, dis] : adj[u]){
            if(dis + d < dist[v]){
                counts[v] = counts[u];
                dist[v] = dis+d;
                pq.push({dist[v], v});
            }
            else if(dis + d == dist[v]){
                counts[v]= (counts[v]+counts[u])%mod;
            }
        }
    }
    return counts[n-1];
}


// No of ways to arrive at a destination

int countPaths(int n, vector<vector<int>>& roads){
    int mod = 10e9 + 7;
    vector<vector<pair<int, long long>>> adj(n);

    for(auto& itr : roads){
        adj[itr[0]].push_back({itr[1], itr[2]});
        adj[itr[1]].push_back({itr[0], itr[2]});
    }
    vector<long long> dist(n, LLONG_MAX);
    vector<int> path(n, 0);
    priority_queue<pair<long long, int>, vector<pair<long long, int>>, greater<>> pq;
    pq.push({0, 0});
    dist[0] = 0;
    path[0] = 1;

    while(!pq.empty()){
        auto [d, u] = pq.top(); pq.pop();

        if(d>dist[u]) continue;

        for(auto& [v, e] : adj[u]){
            if(e+d<dist[v]){
                dist[v] = e+d;
                pq.push({dist[v], v});
                path[v] = path[u];
            } else if(e+d == dist[v]){
                path[v] = (path[v]+path[u])%mod;
            }
        }
    }
    return path[n-1];
}


// Valid Tree

bool dfs(int node, int par, unordered_set<int>& visit, vector<vector<int>>& adj){
    if(visit.count(node)){
        return false;
    }

    visit.insert(node);
    for(auto nei : adj[node]){
        if(nei == par){
            continue;
        }    
        if(!dfs(nei, node, visit, adj)){
            return false;
        }
    }
    return true;
}

bool validTree(int n, vector<vector<int>>& edges){
    if(edges.size()>n){
        return false;
    }
    vector<vector<int>> adj(n);
    for(auto& itr : edges){
        adj[itr[0]].push_back(itr[1]);
        adj[itr[1]].push_back(itr[0]);
    }

    unordered_set<int> visit;
    if(!dfs(0, -1, visit, adj)){
        return false;
    }

    return visit.size() == n;
}


// DP

// Longest Increasing Subsequence

class Solution {
public: 
    int lengthOfLIS(vector<int>& nums){
        int n = nums.size();
        vector<int> LIS(n+1, 1);

        for(int i = 0; i<n; i++){
            for(int j = 0; j<i; j++){
                if(nums[i] > nums[j]){
                    LIS[i] = max(LIS[i], 1 + LIS[j]);
                }
            }
        }

        int ans = 0;
        for(int i = 0; i<n; i++){
            ans = max(ans, LIS[i]);
        }
    }
};


// Longest Increasing Subsequence

int lengthOfLIS(vector<int>& nums){
    int n = nums.size();
    vector<int> LIS(n+1, 1);

    for(int i = 0; i<n; i++){
        for(int j = 0; j<i; j++){
            if(nums[i] > nums[j]){
                LIS[i] = max(LIS[i], 1 + LIS[j]);
            }
        }
    }

    int ans = 0;
    for(int i = 0; i<n; i++){
        ans = max(ans, LIS[i]);
    }
}

// Prim's Algorithm

int spanningTree(int V, vector<vector<int>> adj[]){
    priority_queue<pair<int, int>, vector<pair<int, int>>, greater<pair<int, int>>> pq;

    vector<int> vis(V, 0);
    // {wt, node}

    pq.push({0, 0});
    int sum = 0;
    while(!pq.empty()){
        auto it = pq.top();
        pq.pop();
        int node = it.second;
        int wt = it.first;

        if(vis[node] == 1) continue;

        vis[node] = 1;
        sum += wt;
        for(auto it: adj[node]){
            int adjNode = it[0];
            int edW = it[1];
            if(!vis[adjNode]){
                pq.push({edW, adjNode});
            }
        }
    }
    return sum;
}

int Connection(int n, vector<vector<int>>& connections){
    vector<vector<pair<int, int>>> adj(n);
    for(auto [x, y, z] : connections){
        adj[x].push_back({y, z});
        adj[y].push_back({x, z});
    }

    priority_queue<pair<int, int>, vector<pair<int, int>>, greater<pair<int, int>>> pq;
    vector<int> vis(n, 0);
    int sum = 0;
    for(int i = 0; i<n; i++){
        if(vis[i]){
            continue;
        }
        pq.push({0, i});
        while(!pq.empty()){
            auto [wt, u] = pq.top(); pq.pop();
            if(vis[u]) continue;
            vis[u] = 1;
            sum+=wt;
            for(auto [v, eWt] : adj[u]){
                pq.push({eWt, v});
            }
        }
    }
    return sum;
}


// DSU
class DisjointSet{
    vector<int> parent;
public:
    DisjointSet(int n): parent(n){
        parent.assign(n, -1);
    }

    int find(int node, vector<int>& parent){
        if(parent[node]<0){
            return node;
        }
        return parent[node] = find(parent[node], parent);
    }

    bool union_by_weight(int u, int v, vector<int>& parent){
        int pu = find(u, parent), pv = find(v, parent);

        if(pu == pv){
            return 0;
        }

        if(parent[pu] < parent[pv]){
            parent[pu] += parent[pv];
            pv = parent[pu];
        } else{
            parent[pv] += parent[pu];
            pu = parent[pv];
        }
        return 1;
    }
};

class DSU{
    vector<int> parent;
public:
    DSU(int n): parent(n){
        parent.assign(n, -1);
    }

    int find(int u){
        if(parent[u] < 0){
            return u;
        }

        return parent[u] = find(parent[u]);
    }

    bool union_by_weight(int u, int v){
        int pu = parent[u], pv = parent[v];

        if(pu == pv){
            return 0;
        }
        if(parent[pu] < parent[pv]){
            parent[pu] += parent[pv];
            parent[pv] = pu;
        } else{
            parent[pv] += parent[pu];
            parent[pu] = pv;
        }
        return 1;
    }
};

// Sliding Window
// Fixed Size Window

class Solution {
public:
    int countGoodSubstrings(string s){
        int count = 0;
        int n= s.size();

        int start = 0, end = 0;

        while(end < n){
            if(end-start+1 == 3){
                char a = s[start], b = s[start+1], c = s[end];
                if(a != b && b != c && a != c){
                    count++;
                }
                start++;
            }
            end++;
        }
        return count;
    }
};

// Tree DP = DFS + store results of children + combine for parent

class Solution{
public:
    vector<int> parent, size;
    int n, m;

    int getId(int x, int y){
        return x * m + y;
    }

    int find_set(int u){
        if(parent[u] != u) parent[u] = find_set(parent[u]);
        return parent[u];
    }

    int union_set(int u, int v){
        int u = find_set(u);
        int v = find_set(v);

        if(parent[u] == parent[v]) return false;
        if(size[u] < size[v]) swap(u, v);
        parent[v] = u;
        size[u] += size[v];
        return true;
    }

    vector<int> numIsIslandII(int nRows, int mCols, vector<pair<int, int>>& positions){
        int n = nRows;
        int m = mCols;

        parent.resize(n*m, -1);
        size.resize(n*m, 1);
        vector<int> ans;
        int islands = 0;

        int dx[] = {-1,-1,-1,0,0,1,1,1};
        int dy[] = {-1,0,1,-1,1,-1,0,1};

        for(auto &pos : positions){
            int x = pos.first;
            int y = pos.second;
            int id = getId(x, y);

            if(parent[id] != -1){
                ans.push_back(islands);
                continue;
            }
            parent[id] = id;
            size[id] = 1;
            islands++;

            for(int dir = 0; dir<8; dir++){
                int nx = x + dx[dir];
                int ny = y + dy[dir];
                if(nx<0 || ny<0 || nx>=n || ny>=m) continue;
                int nid = getId(nx, ny);
                if(parent[nid != -1]){
                    if(union_set(id, nid)) islands--;
                }
            }
            ans.push_back(islands);
        }
        return ans;
    }
};


class Solution {
public:
    vector<int> count, res;
    vector<vector<int>> adj;
    int N;

    void dfs1(int u, int parent){
        count[u] = 1;
        for(int v : adj[u]){
            if(v == parent) continue;
            dfs1(v, u);
            count[u] += count[v];
            res[0] += count[v];
            dfs1(v, u);
            count[u] += count[v];
            res[0] += count[v];
        }
    }

    void dfs2(int u, int parent){
        for(int v : adj[u]){
            if(v == parent) continue;
            res[v] == res[u] + (N-count[v]) - count[v];
            dfs2(v, u);
        }
    }

    vector<int> sumOfDistancesInTree(int n, vector<vector<int>>& edges){
        N = n;
        adj.resize(n);
        count.resize(n);
        res.resize(n);

        for(auto& e : edges){
            adj[e[0]].push_back(e[1]);
            adj[e[1]].push_back(e[0]);
        }

        dfs1(0, -1);
        dfs2(0, -1);

        return res;
    }
};


class Solution {
    vector<int> res, count;
    vector<vector<int>> adj;
    int N;

    void dfs1(int u, int parent){
        for(auto v : adj[u]){
            if(v == parent) continue;
            dfs1(v, u);
            count[u]  += count[v];
            res[0] += count[v];
        }
        return;
    }

    void dfs2(int u, int parent){
        for(int v : adj[u]){
            if(v == parent) continue;
            res[v] = res[u] + N - 2*count[v];
            dfs2(v, u);
        }
        return;
    }

    vector<int> sumOfDistancesInTree(int n, vector<vector<int>>& edges){
        N = n;
        adj.resize(n);
        res.resize(n);
        count.resize(n);

        for(auto& e : edges){
            adj[e[0]].push_back(e[1]);
            adj[e[1]].push_back(e[0]);
        }

        dfs1(0, -1);
        dfs2(0, -1);
        return res;
    }
};


class Solution {
public: 
    void dfs(TreeNode* root, int path, vector<int>& res){
        if(!root->left && !root->right){
            res.push_back(path*10 + root->val);
            return;
        }

        if(root->left) dfs(root->left, path*10 + root->val, res);
        if(root->right) dfs(root->right, path*10 + root->val, res);
        return;
    }

    int sumNumbers(TreeNode* root){
        vector<int> res;
        dfs(root, 0, res);
        int sum = 0;
        for(int itr : res){
            sum += itr;
        }
        return sum;
    }
};

class Solution{
    int tilt = 0;

    int dfs(TreeNode* root){
        if(root == nullptr) return 0;

        int left = dfs(root->left);
        int right = dfs(root->right);

        tilt += abs(left-right);
        return root->val + left+right;
    }

    int findTilt(TreeNode* root){
        dfs(root);
        return tilt;
    }
};

class Solution {
public:
    int lengthOfLIS(vector<int>& nums){

        int n = nums.size();

        vector<int> LIS(n+1, 1);

        for(int i = 0; i<n; i++){
            for(int j = 0; j<i; j++){
                if(nums[i] > nums[j]){
                    LIS[i] = max(LIS[i], 1 + LIS[j]);
                }
            }
        }

        int ans = 0;
        for(int i = 0; i<n; i++){
            ans = max(ans, LIS[i]);
        }
        return ans;
    }
};


class Solution {
public:
    int lengthOfLIS(vector<int>& nums){
        int n = nums.size();
        vector<int> LIS(n+1, 1);
        
        for(int i= 0; i<n; i++){
            for(int j = 0; j<i; j++){
                if(nums[i] > nums[j])
                    LIS[i] = max(LIS[i], 1 + LIS[j]);
            }
        }

        int ans = 0;
        for(int i = 0; i<n; i++){
            ans = max(ans, LIS[i]);
        }
        
        return ans;
    }
};

int nums[] = {10, 9, 2, 5, 3, 7, 101, 18};
int DPIS[] = {1, 1, 1, 2, 2, 3, 4, 4};

class Solution{
public:
    int lengthOfLIS(vector<int>& nums){
        int n = nums.size();

        vector<int> dp(n+1, 1);

        for(int i = 0; i<n; i++){
            for(int j = 0; j<i; j++){
                if(nums[i] > nums[j]){
                    dp[i] = max(dp[i], dp[j]+1);
                }
            }
        }
        
        return std*::max_element(dp.begin(), dp.end());
    }
};


int MAX_VAL = 100005;
int phi[MAX_VAL];

void preCompute(){
    for(int i = 0; i< MAX_VAL; i++){
        phi[i] = i;
    }

    for(int i = 2; i < MAX_VAL; i++){
        if(phi[i] = i){
            for(int j= i; j < MAX_VAL; j += i){
                phi[j] -= phi[j]/i;
            }
        }
    }
}

vector<int> coprimeCount(vector<int> A){
    static bool computed = false;
    if(!computed){
        preCompute();
        computed = true;
    }

    vector<int> result;
    for(int num : A){
        result.push_back(phi[num]);
    }
    return result;
}


