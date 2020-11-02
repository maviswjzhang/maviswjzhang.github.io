---
layout:     post
title:     链栈的基本运算
subtitle:  
date:       2020-04-15
author:     Mavis
header-img: img/博客背景.jpg
catalog: true
tags:
    - 数据结构与算法


---

## 链式栈的定义
```c
#include <malloc.h>

typedef char ElemType;

typedef struct lsnode 
{	
	ElemType data;			/*存储结点数据*/
	struct lsnode *next;	/*指针域*/
} LinkStack;
```

## 初始化栈
```c
void InitStack(LinkStack *&ls)		/*ls为引用型参数*/
{
	ls=NULL;
}
```

## 进栈运算
```c
void Push(LinkStack *&ls,ElemType x)		/*进栈运算,ls为引用型参数*/
{
	LinkStack *p;
	p=(LinkStack *)malloc(sizeof(LinkStack));
	p->data=x;
	p->next=ls;
	ls=p;
}
```

## 出栈运算
```c
int Pop(LinkStack *&ls,ElemType &x)		/*出栈运算,ls为引用型参数*/
{
	LinkStack *p;
	if (ls==NULL) 	/*栈空,下溢出*/
		return 0;
	else 
	{
		p=ls;
		x=p->data;
		ls=p->next;
		free(p);
       	return 1;
	}
}
```

## 取栈顶元素运算
```c
int GetTop(LinkStack *ls,ElemType &x)	/*取栈顶元素运算*/
{
	if (ls==NULL) 	/*栈空,下溢出*/
		return 0;
	else 
	{
		x=ls->data;
       	return 1;
   	}
}
```

## 判断栈空运算
```c
int StackEmpty(LinkStack *ls)	/*判断栈空运算*/
{
	if (ls==NULL)
		return 1;
	else
		return 0;
} 
```

## main
```c
void main()
{
	LinkStack *ls;
	ElemType e;
	InitStack(ls);
	printf("栈%s\n",(StackEmpty(ls)==1?"空":"不空"));
	printf("a进栈\n");Push(ls,'a');
	printf("b进栈\n");Push(ls,'b');
	printf("c进栈\n");Push(ls,'c');
	printf("d进栈\n");Push(ls,'d');
	printf("栈%s\n",(StackEmpty(ls)==1?"空":"不空"));
	GetTop(ls,e);
	printf("栈顶元素:%c\n",e);
	printf("出栈次序:");
	while (!StackEmpty(ls))
	{
		Pop(ls,e);
		printf("%c ",e);
	}
	printf("\n");
}
```
