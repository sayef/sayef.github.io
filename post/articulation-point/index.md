# Artuculation Point


![img](https://rawgit.com/sayef/tech/master/uploads/2012/11/art-points.jpg)

If a graph becomes disconnected after removal of any node (and its adjacent edges also) from the graph, then we call that node as an articulation point of the graph. In the picture, we can see that if the node d is removed, the graph is separated into two components. So it's an articulation point. In other words, node v is an articulation point, if there are two such nodes u and w, v node is present in every possible paths from u to v. And the graph that does not have any articulation point, we call that a biconnected graph.

![img](https://rawgit.com/sayef/tech/master/uploads/2012/11/bcntd.jpg)

How to find these points? One naive approach could be like following-

1. Remove a node

2. Run DFS or BFS to check if the graph is connected or not, if not then it's an articulation point.

3. Repeat 1 and 2 for all nodes.

Complexity is O(V(V+E)). There is an efficient solution to this problem which is of O(V+E) complexity.

To achieve this complexity we need to run a mandatory DFS. Let's say, we have connected graph G. For each node v ∈ V : we will declare two matrices i.e. discovery[v], back[v].

discovery[v] is nothing but DFS number. That means, we will assign a number (i.e 1, 2, 3) by increasing order to each node when DFS traverses the graph. First, we will initialize back[v] by discovery[v]. Later on, while the graph will be traversed and backtracked we will change the value of back[v].

Let's say v is a node while traversing the graph. There are two cases. If v is source from where we would like to start our traversal or v is any other node.

- Now the source is an articulation point if it has more than one child.
- For other nodes, v is an articulation point if it has child node w and back[w] >= discovery[v].

We will illustrate these abstract conditions by simulation. Before that we can have a look into the pseudocode.

```
Set visit[] array as unvisited for all node;
Set artpoint[] boolean array as false;
Set predfn = 0;
Set child_of_root = 0;
Call DFS (root);
DFS(v)
{
Visit[v] = true;
artpoint[v] = false;
predfn = predfn + 1;
	discovery[v] = back[v] = predfn;
	for each edge (v, w) ∈ All Edges
		if (v, w) is a tree edge then
			DFS(w);
		    if  v   =  root then child_of_root++;
				if child_of_root = 2 then
					 artpoint[root] = true;
		    else
		     	if back[w] < discovery[v]               //Red Part
		        	back[v] = min (back[v], back[w]);
		     	else if back[w]>= discovery[v] then
                        artpoint[v]=true;
		else if (v, w) is is a back edge then
				back[v] = min (back[v], discovery[w]);   //Blue part
}
```

<script src="http://ideone.com/e.js/7jCJ8Z" type="text/javascript" ></script>

![img](https://rawgit.com/sayef/tech/master/uploads/2012/11/case.jpg)

![img](https://rawgit.com/sayef/tech/master/uploads/2012/11/step1.jpg)

![img](https://rawgit.com/sayef/tech/master/uploads/2012/11/step21.jpg)

![img](https://rawgit.com/sayef/tech/master/uploads/2012/11/step3.jpg)

![img](https://rawgit.com/sayef/tech/master/uploads/2012/11/step4.jpg)

Thus, traversing the whole graph we get four articulation points B, E, F and D.

Java Implementation: http://sfuacm.wikia.com/wiki/Cut_Vertices_(Articulation_Points)

Simple c++ implementation: <http://ideone.com/7jCJ8Z> (input termination system is EOF and i just printed either a node is articulation point or not, they may be repeatedly printed )

