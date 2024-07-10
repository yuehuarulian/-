#include <iostream>
#include <string>
#include <vector>
#include <map>
using namespace std;
class State {
public:
    int a, b, c;
    State(int a, int b, int c) : a(a), b(b), c(c) {}

    bool is_legal()
    {
        return (a == b || b == 0 || b == 3) && (a >= 0 && a <= 3) && (b >= 0 && b <= 3);
    }
    State next_state(int a, int b)
    {
        State ret(*this);
        if (ret.c == 0) {//如果船在左
            ret.a -= a;
            ret.b -= b;
            ret.c = 1;
        }
        else if (ret.c == 1) {//如果在右
            ret.a += a;
            ret.b += b;
            ret.c = 0;
        }
        return ret;
    }
    bool is_goal()
    {
        return (a == 0 && b == 0 && c == 1);
    }
    std::string to_string()
    {
        return "(" + std::to_string(a) + "," + std::to_string(b) + "," + std::to_string(c) + ")";
    }
    friend bool operator==(const State& u, const State& v)
    {
        return u.a == v.a && u.b == v.b && u.c == v.c;
    }
};

vector<State> ans;
map<std::string, bool> vis;
// 五种移动状态
constexpr int da[] = { 0, 0, 1, 1, 2 };
constexpr int db[] = { 1, 2, 0, 1, 0 };
bool dfs(State state, map<std::string, bool> vis0, vector<State> ans0)
{
    if (state.is_goal()) {
        for (State st : ans0) {
            std::cout << st.to_string() << ' ';
        }
        cout << endl << endl;
        return true;
    }
    // 遍历五种移动状态
    for (int i = 0; i < 5; i++) {
        State nxt_state = state.next_state(da[i], db[i]);
        if (!nxt_state.is_legal() || vis0[nxt_state.to_string()])
            continue;
        map<std::string, bool> vis1 = vis0;
        vector<State> ans1 = ans0;
        vis1[nxt_state.to_string()] = true;
        ans1.emplace_back(nxt_state);
        if (dfs(nxt_state, vis1,ans1))
            continue;
    }
    return false;
}

int main()
{
    State init_state = { 3, 3, 0 };
    map<std::string, bool> vis0;
    vector<State> ans0;
    ans0.push_back(init_state);
    vis0[init_state.to_string()] = true;
    dfs(init_state, vis0, ans0);
    return 0;
}