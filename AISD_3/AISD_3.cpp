#include <random>
#include <iostream>
#include <fstream>
#include <chrono>
#include <queue>
using namespace std;



class TreeNode {
public:
	int data;
	TreeNode* left;
	TreeNode* right;
	TreeNode* parent = nullptr;
	
	TreeNode(int value) {
		data = value;
		left = nullptr;
		right = nullptr;
	}
	
};

class Tree {
public:
	TreeNode* root = nullptr;
	
	void TreeInsert(int value) {
		TreeNode* tmp = new TreeNode(value);
		TreeNode* y = nullptr;
		TreeNode* cur = root;
		while (cur) {
			y = cur;
			if (value < cur->data) cur = cur->left;
			else cur = cur->right;
		}
		tmp->parent = y;
		if (!y) root = tmp;
		else if (value < y->data) y->left = tmp;
		else y->right = tmp;
	}

	void preorderPrint(TreeNode* cur)
	{
		if (cur == nullptr)   
		{
			return;
		}
		cout << cur->data << " ";
		preorderPrint(cur->left);   
		preorderPrint(cur->right);  
	}

	void inorderPrint(TreeNode* cur) {
		if (cur == nullptr)
		{
			return;
		}
		inorderPrint(cur->left);
		cout << cur->data << " ";
		inorderPrint(cur->right);
	}

	void postorderPrint(TreeNode* cur) {
		if (cur == nullptr)
		{
			return;
		}
		postorderPrint(cur->left);
		postorderPrint(cur->right);
		cout << cur->data << " ";
	}

	TreeNode* Search(int value) {
		TreeNode* cur = root;
		while (cur and value != cur->data) {
			if (value < cur->data) cur = cur->left;
			else cur = cur->right;
		}
		return cur;
	}

	TreeNode* Min(TreeNode* cur) {
		while (cur->left) cur = cur->left;
		return cur;
	}

	TreeNode* Max(TreeNode* cur) {
		while (cur->right) cur = cur->right;
		return cur;
	}

	TreeNode* Successor(TreeNode* cur) {
		if (cur->right) return Min(cur->right);
		TreeNode* tmp = cur->parent;
		while (tmp and cur == tmp->right) {
			cur = tmp;
			tmp = tmp->parent;
		}
		return tmp;
	}

	TreeNode* Predecessor(TreeNode* cur) {
		if (cur->left) return Max(cur->left);
		TreeNode* tmp = cur->parent;
		while (tmp and cur == tmp->left) {
			cur = tmp;
			tmp = tmp->parent;
		}
		return tmp;
	}

	void Delete(TreeNode* del) {
		TreeNode* y = nullptr;
		TreeNode* x = nullptr;
		if (!del->left or !del->right) y = del;
		else y = Successor(del);
		if (y->left) x = y->left;
		else x = y->right;
		if (x) x->parent = y->parent;
		if (!y->parent) x = root;
		else if (y == y->parent->left) y->parent->left = x;
		else y->parent->right = x;
		del->data = y->data;
	}
	
	int HeightOfTree(TreeNode* node)
	{
		if (node == 0)
			return 0;
		int left, right;
		if (node->left) {
			left = HeightOfTree(node->left);
		}
		else
			left = -1;
		if (node->right) {
			right = HeightOfTree(node->right);
		}
		else
			right = -1;
		int max = left > right ? left : right;
		return max + 1;

	}

	void hght(TreeNode* cur) {
		if (cur == nullptr)
		{
			//cout << -1;
			return;
		}
		if (cur->left == nullptr and cur->right == nullptr) {
			int k = 0;
			auto* y = cur;
			while (y) {
				k++;
				y = y->parent;
			}
			cout << k - 1 << " ";
			return;
		}
		hght(cur->left);
		hght(cur->right);
	}
};

class rbnode {
public:
	int data = -1;
	char color = 'b';
	rbnode* left;
	rbnode* right;
	rbnode* parent;
};

class rbTree {
public:
	rbnode* root = nullptr;

	void left_rotate(rbnode* x) {
		rbnode* y = x->right;
		x->right = y->left;
		if (y->left) y->left->parent = x;
		y->parent = x->parent;
		if (!x->parent) root = y;
		else if (x == x->parent->left) x->parent->left = y;
		else x->parent->right = y;
		y->left = x;
		x->parent = y;
	}

	void right_rotate(rbnode* y) {
		rbnode* x = y->left;
		y->left = x->right;
		if (x->right) x->right->parent = y;
		x->parent = y->parent;
		if (!y->parent) root = x;
		else if (y == y->parent->right) y->parent->right = x;
		else x->parent->left = x;
		x->right = y;
		y->parent = x;
	}

	double rb_insert_fixup(rbnode* z) {
		auto start = chrono::steady_clock::now();
		rbnode* y;
		while (z->parent and z->parent->color == 'r') {

			if (z->parent == z->parent->parent->left) {
				y = z->parent->parent->right;
				if (y and y->color == 'r') {

					z->parent->color = 'b';
					y->color = 'b';
					z->parent->parent->color = 'r';
					z = z->parent->parent;
				}
				else {

					if (z == z->parent->right) {
						z = z->parent;
						left_rotate(z);
					}
					z->parent->color = 'b';
					z->parent->parent->color = 'r';
					right_rotate(z->parent->parent);
				}
			}
			else {
				
				y = z->parent->parent->left;
				//if (z->parent->parent->left == nullptr) y = new rbnode;
				if (y and y->color == 'r') {

					z->parent->color = 'b';
					y->color = 'b';
					z->parent->parent->color = 'r';
					z = z->parent->parent;
				}
				else {
	
					if (z == z->parent->left) {
						z = z->parent;
						right_rotate(z);
					}
					z->parent->color = 'b';
					z->parent->parent->color = 'r';
					left_rotate(z->parent->parent);
				}
			}
			//else break;
		}
		root->color = 'b';
		auto end = chrono::steady_clock::now();
		chrono::duration<double> tm = end - start;
		return tm.count();
	}

	double rb_insert(int value) {
		rbnode* z = new rbnode;
		z->data = value;
		rbnode* y = nullptr;
		rbnode* x = root;
		while (x) {
			y = x;
			if (value < x->data) x = x->left;
			else x = x->right;
		}
		z->parent = y;
		if (!y) root = z;
		else if (z->data < y->data) y->left = z;
		else y->right = z;
		z->left = nullptr;
		z->right = nullptr;
		z->color = 'r';
		

		double time = rb_insert_fixup(z);

		
		return time;
	}

	void hght(rbnode* cur) {
		if (cur == nullptr)
		{
			return;
		}
		if (cur->left == nullptr and cur->right == nullptr) {
			int k = 0;
			
			auto* y = cur;
			while (y) {
				if (y->color == 'b') k++;
				//k++;
				y = y->parent;
			}
			cout << k - 1 << " ";
			//return;
		}
		hght(cur->left);
		hght(cur->right);
	}

	int HeightOfTree(rbnode* node)
	{
		if (node == 0)
			return 0;
		int left, right;
		if (node->left) {
			left = HeightOfTree(node->left);
		}
		else
			left = -1;
		if (node->right) {
			right = HeightOfTree(node->right);
		}
		else
			right = -1;
		int max = left > right ? left : right;
		return max + 1;

	}

	void preorderPrint(rbnode* cur)
	{
		if (cur == nullptr)
		{
			return;
		}
		cout << cur->data << " ";
		preorderPrint(cur->left);
		preorderPrint(cur->right);
	}

	void inorderPrint(rbnode* cur) {
		if (cur == nullptr)
		{
			return;
		}
		inorderPrint(cur->left);
		cout << cur->data << " ";
		inorderPrint(cur->right);
	}

	void postorderPrint(rbnode* cur) {
		if (cur == nullptr)
		{
			return;
		}
		postorderPrint(cur->left);
		postorderPrint(cur->right);
		cout << cur->data << " ";
	}

	rbnode* Search(int value) {
		rbnode* cur = root;
		while (cur and value != cur->data) {
			if (value < cur->data) cur = cur->left;
			else cur = cur->right;
		}
		return cur;
	}

	rbnode* Min(rbnode* cur) {
		while (cur->left) cur = cur->left;
		return cur;
	}

	rbnode* Max(rbnode* cur) {
		while (cur->right) cur = cur->right;
		return cur;
	}

	rbnode* Successor(rbnode* cur) {
		if (cur->right) return Min(cur->right);
		rbnode* tmp = cur->parent;
		while (tmp and cur == tmp->right) {
			cur = tmp;
			tmp = tmp->parent;
		}
		return tmp;
	}

	rbnode* Predecessor(rbnode* cur) {
		if (cur->left) return Max(cur->left);
		rbnode* tmp = cur->parent;
		while (tmp and cur == tmp->left) {
			cur = tmp;
			tmp = tmp->parent;
		}
		return tmp;
	}

	double RB_Delete(rbnode* z) {
		rbnode* y = nullptr;
		rbnode* x = nullptr;
		if (z->left == nullptr or z->right == nullptr) y = z;
		else y = Successor(z);
		if (y->left) x = y->left;
		else x = y->right;
		if (x) x->parent = y->parent;
		if (!y->parent) root = x;
		else if (y == y->parent->left) y->parent->left = x;
		else y->parent->right = x;
		z->data = y->data;
		auto start = chrono::steady_clock::now();
		if (y->color == 'b') RB_Delete_fixup(x);
		auto end = chrono::steady_clock::now();
		chrono::duration<double> tm = end - start;
		return tm.count();
	}

	void RB_Delete_fixup(rbnode* x) {
		rbnode* w = nullptr;
		while (x and x != root and x->color == 'b') {
			if (x == x->parent->left) {
				w = x->parent->right;
				if (w and w->color == 'r') {
					w->color = 'b';
					x->parent->color = 'r';
					left_rotate(x->parent);
					w = x->parent->right;
				}
				if ((!w->left or w->left->color == 'b') and (!w->right or w->right->color == 'b')) {
					w->color = 'r';
					x = x->parent;
				}
				else {
					if (!w->right or w->right->color == 'b') {
						w->left->color = 'b';
						w->color = 'r';
						right_rotate(w);
						w = x->parent->right;
					}
					w->color = x->parent->color;
					x->parent->color = 'b';
					w->right->color = 'b';
					left_rotate(x->parent);
					x = root;
				}
			}
			else {
				w = x->parent->left;
				if (w and w->color == 'r') {
					w->color = 'b';
					x->parent->color = 'r';
					right_rotate(x->parent);
					w = x->parent->left;
				}
				if ((!w->right or w->right->color == 'b') and (!w->left or w->left->color == 'b')) {
					w->color = 'r';
					x = x->parent;
				}
				else {
					if (!w->left or w->left->color == 'b') {
						w->right->color = 'b';
						w->color = 'r';
						left_rotate(w);
						w = x->parent->left;
					}
					w->color = x->parent->color;
					x->parent->color = 'b';
					w->left->color = 'b';
					right_rotate(x->parent);
					x = root;
				}
			}
		}
		//root->color = 'b';
	}

	int depth(rbnode* node) {
		rbnode* x = node;
		int k = 0;
		while (x) {
			k++;
			x = x->parent;
		}
		return k;
	}

	void wideprint(rbnode* cur, int k) {
		if (!cur) return;
		if (depth(cur) == k) cout << cur->data << " ";
		wideprint(cur->left, k);
		wideprint(cur->right, k);
	}

	void print() {
		rbnode* x = root;
		int h = HeightOfTree(root);
		for (int i = 0; i <= h; i++) {
			wideprint(x, i + 1);
			cout << endl;
		}
	}

};

class avlnode {
public:
	int data = -1;
	int height = 0;
	avlnode* left;
	avlnode* right;
	avlnode* parent;
	double time = 0;
	
};

class avlTree {
public:
	avlnode* root = nullptr;

	unsigned int height(avlnode* p)
	{
		return p ? p->height : 0;
	}

	int bfactor(avlnode* p)
	{
		return height(p->left) - height(p->right);
	}

	double insert_fixup(avlnode* node) {
		auto start = chrono::steady_clock::now();
		int balance = 0;
		avlnode* tmp = node;
		while (tmp) {
			balance = bfactor(tmp);
			//cout << balance << endl;
			if (balance == -2) {
				if (bfactor(tmp->right) > 0)
					rotateright(tmp->right);
				rotateleft(tmp);
			}
			if (balance == 2) {
				if (bfactor(tmp->left) < 0)
					rotateleft(tmp->left);
				rotateright(tmp);
			}
			//cout << bfactor(tmp);
			if (balance > -1 and balance < 1) break;

			tmp = tmp->parent;
		}
		//cout << endl;
		auto end = chrono::steady_clock::now();
		chrono::duration<double> tm = end - start;
		return tm.count();
	}

	void rotateright(avlnode* y)
	{
		avlnode* x = y->left;
		y->left = x->right;
		if (x->right) x->right->parent = y;
		x->parent = y->parent;
		if (!y->parent) root = x;
		else if (y == y->parent->right) y->parent->right = x;
		else x->parent->left = x;
		x->right = y;
		y->parent = x;
		y->height = max(height(y->left),
			height(y->right)) + 1;
		x->height = max(height(x->left),
			height(x->right)) + 1;
	}

	void rotateleft(avlnode* x)
	{
		avlnode* y = x->right;
		x->right = y->left;
		if (y->left) y->left->parent = x;
		y->parent = x->parent;
		if (!x->parent) root = y;
		else if (x == x->parent->left) x->parent->left = y;
		else x->parent->right = y;
		y->left = x;
		x->parent = y;
		x->height = max(height(x->left),
			height(x->right)) + 1;
		y->height = max(height(y->left),
			height(y->right)) + 1;
	}

	avlnode* insert(avlnode* &par, avlnode* &node, int value) {
		if (node == nullptr) {
			node = new avlnode;
			node->parent = par;
			node->data = value;
			node->left = nullptr;
			node->right = nullptr;
			node->height = 1;
			//if (!root) root = node;
			return node;
		}
		if (value < node->data) {
			insert(node, node->left, value);
			node->time = insert_fixup(node);
		}
		if (value > node->data) {
			insert(node, node->right, value);
			node->time = insert_fixup(node);
		}
		node->height = 1 + max(height(node->left), height(node->right));
		return node;
	}

	int HeightOfTree(avlnode* node)
	{
		if (node == 0)
			return 0;
		int left, right;
		if (node->left) {
			left = HeightOfTree(node->left);
		}
		else
			left = -1;
		if (node->right) {
			right = HeightOfTree(node->right);
		}
		else
			right = -1;
		int max = left > right ? left : right;
		return max + 1;

	}

	void inorderPrint(avlnode* cur) {
		if (cur == nullptr)
		{
			return;
		}
		inorderPrint(cur->left);
		cout << cur->data << ' ';
		
		inorderPrint(cur->right);
	}

	void hght(avlnode* cur) {
		if (cur == nullptr)
		{
			return;
		}
		cout << HeightOfTree(cur->right) - HeightOfTree(cur->left) << ' ';
		hght(cur->left);
		hght(cur->right);
	}

	avlnode* Search(int value) {
		avlnode* cur = root;
		while (cur and value != cur->data) {
			if (value < cur->data) cur = cur->left;
			else cur = cur->right;
		}
		return cur;
	}

	avlnode* Min(avlnode* cur) {
		while (cur->left) cur = cur->left;
		return cur;
	}

	avlnode* Max(avlnode* cur) {
		while (cur->right) cur = cur->right;
		return cur;
	}

	avlnode* Successor(avlnode* cur) {
		if (cur->right) return Min(cur->right);
		avlnode* tmp = cur->parent;
		while (tmp and cur == tmp->right) {
			cur = tmp;
			tmp = tmp->parent;
		}
		return tmp;
	}

	avlnode* Predecessor(avlnode* cur) {
		if (cur->left) return Max(cur->left);
		avlnode* tmp = cur->parent;
		while (tmp and cur == tmp->left) {
			cur = tmp;
			tmp = tmp->parent;
		}
		return tmp;
	}

	double del(avlnode* z) {
		avlnode* y = nullptr;
		avlnode* x = nullptr;
		if (z->left == nullptr or z->right == nullptr) y = z;
		else y = Successor(z);
		if (y->left) x = y->left;
		else x = y->right;
		if (x) x->parent = y->parent;
		if (!y->parent) root = x;
		else if (y == y->parent->left) y->parent->left = x;
		else y->parent->right = x;
		z->data = y->data;
		auto start = chrono::steady_clock::now();
		insert_fixup(z);
		auto end = chrono::steady_clock::now();
		chrono::duration<double> tm = end - start;
		return tm.count();
	}

	void wideprint(avlnode* cur, int k) {
		if (!cur) return;
		if (cur->height == k) cout << cur->data << " ";
		wideprint(cur->left, k);
		wideprint(cur->right, k);
	}

	void print() {
		avlnode* x = root;
		int h = HeightOfTree(root);
		for (int i = h; i >= 0; i--) {
			wideprint(x, i + 1);
			cout << endl;
		}
	}

};


void main() {
	fstream f;
	f.open("graph.py", ios::out);
	f << "from matplotlib import pyplot as plt" << endl << endl;
	f << "import numpy as np" << endl;
	f << "from scipy.optimize import curve_fit\n\n";
	f << "def f(x, a) :\n\treturn (a * np.log(x))\n";
	//f << "f = lambda n: 2*np.log(n+1)" << endl;
	//f << "g = lambda n: 1.44*np.log2(n + 2)-0.328" << endl;
	int k = 0;
	int n = 15000;
	int *t_h = new int[n];
	float* rbt_h = new float[n];
	float* avl_h = new float[n];
	Tree t;
	
	rbTree rb;
	
	avlTree avl;
	avlnode* par = nullptr;

	random_device dev;
	mt19937 rng(dev());
	for (int i = 0; i < n; i++) {
		uniform_int_distribution<std::mt19937::result_type> dist6(1, 1000000);
		int x = dist6(rng);
		auto start = chrono::steady_clock::now();
		//rb.rb_insert(x);
		auto end = chrono::steady_clock::now();
		chrono::duration<double> tm = end - start;

		rbt_h[i] = rb.rb_insert(x);
		auto start2 = chrono::steady_clock::now();
		avl.insert(par, avl.root, x);
		auto end2 = chrono::steady_clock::now();
		chrono::duration<double> tm2 = end2 - start2;
		avl_h[i] = avl.Search(x)->time;
		//avl_h[i] = avl.HeightOfTree(avl.root);
		//t.TreeInsert(x);
		//rbt_h[i] = avl_h[i] = tm.count();
		t_h[i] = t.HeightOfTree(t.root);
		cout << i << endl;
	}

	//удаление и обход в ширину
	
	//avl.hght(avl.root);
	cout << endl;
	for (int i = 0; i < n; i++) {
		

		
		if (avl.root) avl_h[i] = avl.del(avl.root);
		cout << i << endl;
		//avl.hght(avl.root);
		//cout << endl;
		//avl_h[i] = avl.del(avl.root);

	}
	
	f << "x = [";
	for (int i = 0; i < n; i++) {
		if (i != n - 1) f << t_h[i] << ',';
		else f << t_h[i] << ']';
	}
	f << endl << "y = np.array([";
	for (int i = 0; i < n; i++) {
		if (i != n - 1) f << rbt_h[i] << ',';
		else f << rbt_h[i] << "])";
	}
	f << endl << "a = np.array([";
	for (int i = 0; i < n; i++) {
		if (i != n - 1) f << avl_h[i] << ',';
		else f << avl_h[i] << "])";
	}
	f << endl << "z = np.array([";
	for (int i = 0; i < n; i++) {
		if (i != n - 1) f << i+1 << ',';
		else f << i + 1 << "])";
	}
	float sum = 0;
	for (int i = 0; i < n; i++) sum += avl_h[i];
	cout << endl << sum / 15000 << endl;
	f << "\nargrb, _ = curve_fit(f, z, y)";
	f << "\nargavl, _ = curve_fit(f, z, a)";
	f << "\nfuncrb = argrb * np.log(z)";
	f << "\nfuncavl = argavl * np.log(z)";
	//f << "\ntime_rbt = (sum(y)/len(y))";
	f << "\ntime_avl = (sum(a)/len(a))";
	f << "\nprint(time_avl)";
	f << "\nplt.xlabel(\"Number of elements\")\nplt.ylabel(\"Time\")";
	//f << endl << "plt.plot(z,funcrb, linewidth=1, label=\"Red-Black tree\")";
	//f << endl << "plt.plot(z,funcavl, linewidth=1, label=\"AVL Binary tree\")";
	f << endl << "plt.scatter(z, a, c='red', s = 0.4, label=\"AVL tree\")";
	//f << endl << "plt.plot(z,x, linewidth=1, label=\"Binary tree\")";
	//f << endl << "n = np.linspace(0," << n << ")";
	//f << endl << "t = f(n)";
	//f << endl << "e = g(n)";
	//f << endl << "plt.plot(n,t, linewidth=1)";
	//f << endl << "plt.plot(n,e, linewidth=1)";
	f << endl << "plt.legend(loc=\"upper left\")";
	f << endl << "plt.show()";
	f.close();
	system("python graph.py");
}
