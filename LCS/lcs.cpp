#include <vector>
using namespace std;

// Longest Common Subsequence
class Solution{
public:
    int lengthOfLCS(vector<int>& text1, vector<int>& text2){
        int n = text1.size(), m = text2.size();
        vector<vector<int>> lis(n+1, vector<int>(m+1, 0));

        for(int i = 1; i<n+1; i++){
            for(int j = 1; j<m+1; j++){
                if(text1[i-1] == text2[j-1]){
                    lis[i][j] = 1 + lis[i-1][j-1];
                } else{
                    lis[i][j] = max(lis[i][j-1], lis[i-1][j]);
                }
            }
        }
        return lis[n+1][m+1];
    }
};