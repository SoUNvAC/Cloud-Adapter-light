// Exact descending FP64 run merge; constant label and log-score shift per run.
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <queue>
#include <stdexcept>
#include <string>
#include <vector>
struct Run{
 std::ifstream f;double value=0,shift=0;int label=0;bool valid=false;std::uint64_t count=0;
 Run(const std::string& p,int y,double b):f(p,std::ios::binary),shift(b),label(y){if(!f)throw std::runtime_error("missing run");f.seekg(0,std::ios::end);auto bytes=f.tellg();if(bytes<0||bytes%8)throw std::runtime_error("invalid FP64 run size");count=bytes/8;f.seekg(0);next();}
 void next(){valid=bool(f.read(reinterpret_cast<char*>(&value),8));if(valid){value-=shift;if(!std::isfinite(value))throw std::runtime_error("nonfinite score");}}
};
struct Node{double v;std::size_t i;bool operator<(const Node& o)const{return v<o.v;}};
int main(int argc,char**argv){try{
 if(argc!=2)throw std::runtime_error("manifest required");std::ifstream mf(argv[1]);if(!mf)throw std::runtime_error("manifest missing");std::string path;int label;double shift;std::vector<Run> runs;long double P=0,N=0;
 while(mf>>path>>label>>shift){if(label!=0&&label!=1)throw std::runtime_error("invalid class");runs.emplace_back(path,label,shift);if(label)P+=runs.back().count;else N+=runs.back().count;}
 std::priority_queue<Node> heap;for(std::size_t i=0;i<runs.size();++i)if(runs[i].valid)heap.push({runs[i].value,i});
 long double tp=0,fp=0,gp=0,gn=0,ap=0,wins=0;double current=0;bool first=true;
 auto flush=[&](){if(first)return;tp+=gp;fp+=gn;if(P)ap+=gp*tp/(tp+fp);if(P&&N)wins+=gp*(N-fp+gn/2);gp=gn=0;};
 while(!heap.empty()){auto v=heap.top();heap.pop();if(first){current=v.v;first=false;}else if(v.v!=current){flush();current=v.v;}auto& r=runs[v.i];if(r.label)++gp;else ++gn;r.next();if(r.valid)heap.push({r.value,v.i});}flush();if(tp!=P||fp!=N)throw std::runtime_error("support mismatch");
 std::cout<<std::setprecision(17)<<"{\"P\":"<<std::uint64_t(P)<<",\"N\":"<<std::uint64_t(N)<<",\"AP\":";if(P&&N)std::cout<<double(ap/P);else std::cout<<"null";std::cout<<",\"AUC\":";if(P&&N)std::cout<<double(wins/(P*N));else std::cout<<"null";std::cout<<"}\n";
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
