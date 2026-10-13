#include <vector>
#include <numeric>
#include <algorithm>
#include <cmath>

struct DSU {
    std::vector<int> parent, size;
    explicit DSU(int n): parent(n),size(n,1) {std::iota(parent.begin(),parent.end(),0);}
    int root(int a) {while(parent[a]!=a){parent[a]=parent[parent[a]];a=parent[a];}return a;}
    bool join(int a,int b){a=root(a);b=root(b);if(a==b)return false;if(size[a]<size[b])std::swap(a,b);parent[b]=a;size[a]+=size[b];return true;}
};

// Exact graphic-matroid rank on the minor formed by mandatory edges.
extern "C" int envelopes(int p,int m,const int *edge,const int *forced,
                         const int *allowed,int *lo,int *hi,int *difference) {
    DSU mandatory(p);int mandatory_count=0;
    std::vector<int> offset(p,0);
    for(int e=0;e<m;++e)if(forced[e]){
        if(!allowed[e] || !mandatory.join(edge[2*e],edge[2*e+1]))return 1;
        ++mandatory_count;++offset[edge[2*e]];++offset[edge[2*e+1]];
    }
    const int residual_rank=p-1-mandatory_count;
    auto rank_of=[&](int i,int j,int filter) {
        DSU forest=mandatory;int rank=0;
        for(int e=0;e<m;++e)if(allowed[e]&&!forced[e]){
            int a=edge[2*e],b=edge[2*e+1];
            int ci=(a==i || b==i),cj=(j>=0 && (a==j || b==j));
            int c=ci-cj;
            bool take=(filter==0?c>0:(filter==1?c>=0:(filter==2?c<0:c<=0)));
            if(take && forest.join(a,b))++rank;
        }
        return rank;
    };
    DSU check=mandatory;int available_rank=0;
    for(int e=0;e<m;++e)if(allowed[e]&&!forced[e]&&check.join(edge[2*e],edge[2*e+1]))++available_rank;
    if(available_rank!=residual_rank)return 2;
    for(int i=0;i<p;++i){
        hi[i]=offset[i]+rank_of(i,-1,0);
        lo[i]=offset[i]+residual_rank-rank_of(i,-1,3);
    }
    for(int i=0;i<p;++i)for(int j=0;j<p;++j){
        if(i==j){difference[i*p+j]=0;continue;}
        difference[i*p+j]=offset[i]-offset[j]+residual_rank-rank_of(i,j,2)-rank_of(i,j,3);
    }
    return 0;
}

// Exact tau-a numerator via rank preprocessing and Fenwick inversion counts.
// Deferred updates within an x-tie group exclude tied-x pairs. The joint tie
// correction also handles constant or discrete margins without tau-b scaling.
extern "C" int kendall_tau_a(int n,int p,const double *x,double *out) {
    if(n<2)return 1;
    std::vector<std::vector<int>> order(p,std::vector<int>(n)),rank(p,std::vector<int>(n));
    std::vector<long long> ties(p,0);
    for(int j=0;j<p;++j){
        std::iota(order[j].begin(),order[j].end(),0);
        std::stable_sort(order[j].begin(),order[j].end(),[&](int a,int b){return x[a*p+j]<x[b*p+j];});
        int level=0;
        for(int start=0;start<n;){
            int stop=start+1;while(stop<n && x[order[j][stop]*p+j]==x[order[j][start]*p+j])++stop;
            long long count=stop-start;ties[j]+=count*(count-1)/2;
            ++level;for(int k=start;k<stop;++k)rank[j][order[j][k]]=level;
            start=stop;
        }
    }
    long long total=static_cast<long long>(n)*(n-1)/2;int edge=0;
    for(int i=0;i<p;++i)for(int j=i+1;j<p;++j){
        std::vector<int> fenwick(n+1,0),joint(n+1,0);long long discordant=0,joint_ties=0;int inserted=0;
        for(int start=0;start<n;){
            int stop=start+1;while(stop<n && x[order[i][stop]*p+i]==x[order[i][start]*p+i])++stop;
            for(int k=start;k<stop;++k){
                int value=rank[j][order[i][k]],prefix=0;
                for(int a=value;a>0;a-=a&-a)prefix+=fenwick[a];
                discordant+=inserted-prefix;joint_ties+=joint[value];++joint[value];
            }
            for(int k=start;k<stop;++k){
                int value=rank[j][order[i][k]];joint[value]=0;
                for(int a=value;a<=n;a+=a&-a)++fenwick[a];
            }
            inserted+=stop-start;start=stop;
        }
        out[edge++]=static_cast<double>(total-ties[i]-ties[j]+joint_ties-2*discordant)/total;
    }
    return 0;
}

// Rowwise order-two kernel averages, including ties, in O(p^2 n log n).
// A forward/backward sweep counts concordances minus discordances against
// observations with strictly smaller/larger x. Updates are deferred at ties.
extern "C" int kendall_row_means(int n,int p,const double *x,double *out) {
    if(n<2)return 1;
    const int m=p*(p-1)/2;
    std::vector<std::vector<int>> order(p,std::vector<int>(n)),rank(p,std::vector<int>(n));
    for(int j=0;j<p;++j){
        std::iota(order[j].begin(),order[j].end(),0);
        std::stable_sort(order[j].begin(),order[j].end(),[&](int a,int b){return x[a*p+j]<x[b*p+j];});
        int level=0;
        for(int start=0;start<n;){
            int stop=start+1;while(stop<n&&x[order[j][stop]*p+j]==x[order[j][start]*p+j])++stop;
            ++level;for(int k=start;k<stop;++k)rank[j][order[j][k]]=level;
            start=stop;
        }
    }
    std::fill(out,out+static_cast<long long>(n)*m,0.0);
    int edge=0;
    for(int i=0;i<p;++i)for(int j=i+1;j<p;++j,++edge){
        for(int direction=0;direction<2;++direction){
            std::vector<int> fenwick(n+1,0);int inserted=0;
            auto prefix=[&](int r){int ans=0;for(;r>0;r-=r&-r)ans+=fenwick[r];return ans;};
            int start=0;
            while(start<n){
                int stop=start+1;
                auto ix=[&](int k){return order[i][direction?n-1-k:k];};
                while(stop<n&&x[ix(stop)*p+i]==x[ix(start)*p+i])++stop;
                for(int k=start;k<stop;++k){
                    int row=ix(k),r=rank[j][row];
                    int below=prefix(r-1),above=inserted-prefix(r);
                    out[static_cast<long long>(row)*m+edge]+=direction?above-below:below-above;
                }
                for(int k=start;k<stop;++k)for(int r=rank[j][ix(k)];r<=n;r+=r&-r)++fenwick[r];
                inserted+=stop-start;start=stop;
            }
        }
    }
    for(long long k=0;k<static_cast<long long>(n)*m;++k)out[k]/=(n-1);
    return 0;
}

// Streaming maximum: no B-by-K contrast matrix is materialized.
extern "C" int max_edge_contrast(int B,int m,int K,const double *draws,
    const double *score,const int *a,const int *b,const double *scale,
    int centered,double *out) {
    for(int k=0;k<K;++k)if(a[k]<0||a[k]>=m||b[k]<0||b[k]>=m||scale[k]<=0)return 1;
    for(int row=0;row<B;++row){
        const double *d=draws+static_cast<long long>(row)*m;double maximum=0;
        for(int k=0;k<K;++k){
            double value=d[a[k]]-d[b[k]];
            if(!centered)value-=score[a[k]]-score[b[k]];
            maximum=std::max(maximum,std::abs(value)/scale[k]);
        }
        out[row]=maximum;
    }
    return 0;
}
