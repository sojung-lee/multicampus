package day11;

public class Outer {
	
		   int a=1;
	static int b=2;
	
	void print() {
		Inner i=new Inner();
		System.out.println(i.c);
		
		Outer.Inner oi=this.new Inner();
		System.out.println(oi.c);
		
	}
	
	class Inner{ //Inner Member class - non-static
		int c=3;
		void sub() {
			System.out.println("sub()***");
		}
	}//Inner////////
	
	static class SInner{
			   int d=4;
		static int e=5;
		void foo() {
			System.out.println("foo()###");
		}
		static void bar() {
			System.out.println("bar()@@@");
		}
	}//SInner///////
	

}/////////////////////////////////////
