// Exhaustive labelled-tree audit for 2 <= p <= 8.
// Input: p, q, then q directed strict edge comparisons (stronger, weaker).
// Edges are zero-indexed lexicographic unordered vertex pairs.
// Output: one row for every compatible tree, including its Pruefer code number.
// A tree is compatible iff the strict relation together with all tree-path
// dominance arcs is acyclic. Tree-path arcs alone are bipartite (T -> E\T),
// so every directed cycle contains a strict arc and is inconsistent even
// when the optimality inequalities are weak.
#include <algorithm>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <vector>

int main() {
  int p,q; if(!(std::cin>>p>>q) || p<2 || p>8) return 2;
  int m=p*(p-1)/2, id[8][8];
  std::vector<int> eu,ev;
  for(int i=0;i<p;i++) for(int j=i+1;j<p;j++) {
    id[i][j]=id[j][i]=eu.size(); eu.push_back(i); ev.push_back(j);
  }
  std::vector<uint64_t> base(m,0);
  for(int k=0;k<q;k++) {
    int a,b;std::cin>>a>>b;
    if(a<0||b<0||a>=m||b>=m||a==b) return 3;
    base[a]|=uint64_t(1)<<b;
  }
  uint64_t count=1;for(int i=0;i<p-2;i++)count*=p;
  uint64_t accepted=0;
  for(uint64_t code=0;code<count;code++) {
    uint64_t z=code, mask=0;
    int seq[8],deg[8],work[8];
    std::fill(deg,deg+p,1);
    for(int k=0;k<p-2;k++){seq[k]=z%p;z/=p;deg[seq[k]]++;}
    std::copy(deg,deg+p,work);
    std::vector<std::pair<int,int>> adj[8];
    auto add=[&](int i,int j){int e=id[i][j];mask|=uint64_t(1)<<e;
      adj[i].push_back({j,e});adj[j].push_back({i,e});};
    for(int k=0;k<p-2;k++){
      int leaf=0;while(work[leaf]!=1)leaf++;
      add(leaf,seq[k]);work[leaf]--;work[seq[k]]--;
    }
    int a=0;while(work[a]!=1)a++;int b=a+1;while(work[b]!=1)b++;add(a,b);
    auto arcs=base;
    for(int root=0;root<p;root++){
      int parent[8],pe[8],queue[8],head=0,tail=1;
      std::fill(parent,parent+p,-1);parent[root]=root;queue[0]=root;
      while(head<tail){int v=queue[head++];for(auto [u,e]:adj[v])if(parent[u]<0){
        parent[u]=v;pe[u]=e;queue[tail++]=u;
      }}
      for(int end=root+1;end<p;end++){
        int e=id[root][end];if(mask&(uint64_t(1)<<e))continue;
        for(int v=end;v!=root;v=parent[v])arcs[pe[v]]|=uint64_t(1)<<e;
      }
    }
    int indegree[28]={0};
    for(int e=0;e<m;e++)for(uint64_t bits=arcs[e];bits;bits&=bits-1)indegree[__builtin_ctzll(bits)]++;
    int queue[28],head=0,tail=0;
    for(int e=0;e<m;e++)if(!indegree[e])queue[tail++]=e;
    while(head<tail){int e=queue[head++];for(uint64_t bits=arcs[e];bits;bits&=bits-1){
      int f=__builtin_ctzll(bits);if(!--indegree[f])queue[tail++]=f;
    }}
    if(tail!=m)continue;
    accepted++;
    std::cout<<code<<','<<mask;
    for(int i=0;i<p;i++)std::cout<<','<<deg[i];
    for(int i=0;i<p;i++){
      int lower=1,upper=p;
      for(int j=0;j<p;j++){lower+=deg[j]>deg[i];upper-=deg[j]<deg[i];}
      std::cout<<','<<lower<<','<<upper;
    }
    std::cout<<'\n';
  }
  std::cerr<<"enumerated="<<count<<" compatible="<<accepted<<'\n';
}
