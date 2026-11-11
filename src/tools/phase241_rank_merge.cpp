// Exact descending-score tie-aware AP/AUC on lossless disk-sorted scene runs.
// No binning, quantization, or sampling. Buffer size independent of pool length.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <queue>
#include <stdexcept>
#include <string>
#include <vector>
struct Run {
 std::ifstream scores, labels; double score=0; unsigned char label=0;
 Run(const std::string& s,const std::string& y):scores(s,std::ios::binary),labels(y,std::ios::binary) {next();}
 bool valid=false;
 void next(){valid=bool(scores.read(reinterpret_cast<char*>(&score),8));if(valid){if(!labels.read(reinterpret_cast<char*>(&label),1))throw std::runtime_error("truncated labels");if(!std::isfinite(score)||label>1)throw std::runtime_error("bad record");}}
};
struct Node{double score;size_t run;bool operator<(const Node& x)const{return score<x.score;}};
int main(int argc,char**argv){try{
 if(argc==3 && std::string(argv[1])=="threshold"){
  std::ifstream mf(argv[2]);std::string s,y;std::uint64_t n;std::vector<Run> runs;std::vector<std::uint64_t> denom;
  while(mf>>s>>y>>n){if(n){runs.emplace_back(s,y);denom.push_back(n);}}
  if(runs.empty())throw std::runtime_error("No validation negatives");
  std::priority_queue<Node> heap;for(size_t i=0;i<runs.size();++i)if(runs[i].valid)heap.push({runs[i].score,i});
  std::vector<std::uint64_t> accepted(runs.size(),0),group(runs.size(),0);double current=heap.top().score;
  auto rate=[&](bool withgroup){long double v=0;for(size_t i=0;i<runs.size();++i)v+=(long double)(accepted[i]+(withgroup?group[i]:0))/denom[i];return v/runs.size();};
  auto emit=[&](){std::cout<<std::setprecision(17)<<"{\"threshold\":"<<std::nextafter(current,INFINITY)<<",\"boundary_score\":"<<current<<",\"mean_val_FPR\":"<<double(rate(false))<<",\"inclusive_boundary_FPR\":"<<double(rate(true))<<",\"negative_scenes\":"<<runs.size()<<"}\n";};
  while(!heap.empty()){
   auto node=heap.top();heap.pop();
   if(node.score!=current){if(rate(true)>.01L){emit();return 0;}for(size_t i=0;i<runs.size();++i){accepted[i]+=group[i];group[i]=0;}current=node.score;}
   ++group[node.run];runs[node.run].next();if(runs[node.run].valid)heap.push({runs[node.run].score,node.run});
  }
  if(rate(true)>.01L){emit();return 0;}throw std::runtime_error("Unexpected all-negative acceptance");
 }
 if(argc!=4)throw std::runtime_error("manifest positive_count negative_count");
 const long double P=std::stoull(argv[2]),N=std::stoull(argv[3]);
 std::ifstream mf(argv[1]);std::string s,y;std::vector<Run> runs;
 while(mf>>s>>y)runs.emplace_back(s,y);
 std::priority_queue<Node> heap;for(size_t i=0;i<runs.size();++i)if(runs[i].valid)heap.push({runs[i].score,i});
 long double tp=0,fp=0,ap=0,auc=0,gp=0,gn=0;double current=0;bool first=true;unsigned long long ties=0;
 auto flush=[&](){if(first)return;tp+=gp;fp+=gn;if(P>0)ap+=gp*tp/(tp+fp);if(P>0&&N>0)auc+=gp*(N-fp+gn/2);gp=gn=0;++ties;};
 while(!heap.empty()){
  auto n=heap.top();heap.pop();if(first){current=n.score;first=false;}else if(n.score!=current){flush();current=n.score;}
  auto& r=runs[n.run];if(r.label)++gp;else ++gn;r.next();if(r.valid)heap.push({r.score,n.run});
 }
 flush();if(tp!=P||fp!=N)throw std::runtime_error("support mismatch");
 std::cout<<std::setprecision(17)<<"{\"AP\":";
 if(P>0&&N>0)std::cout<<double(ap/P);else std::cout<<"null";
 std::cout<<",\"AUC\":";if(P>0&&N>0)std::cout<<double(auc/(P*N));else std::cout<<"null";
 std::cout<<",\"positive\":"<<std::uint64_t(P)<<",\"negative\":"<<std::uint64_t(N)<<",\"unique_scores\":"<<ties<<"}\n";
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;} }
