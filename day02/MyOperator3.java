package day02;

public class MyOperator3 {

	public static void main(String[] args) {
		System.out.println("6. 조건연산자(삼항연산자)----");
		/* 변수 선언문 =(조건식)? 값1:값2;
		 * => 조건식이 true이면 값1을 변수에 대입.
		 * 			 false이면 값2를 변수에 대입한다.
		 * 이때 주의. 변수와 값1, 값2는 같은 자료형이어야 한다
		 * */
		int a=50;
		int b=60;
		int result=(a<b)?a:b;
		System.out.println("result="+result);
		
		String str=(a>b)? a+"":b+"";
		System.out.println("str="+str);
		
		System.out.println("7.할당연산자(연산후 대입연산자)---");
		/* =, +=, -=, *=, /=, %=, >>=, <<=,>>>=, &=, |=, ^=
		 * */
		int i=1;
		i+=3; // i = i+3;
		System.out.println("i="+i);
		
		int j=15;
		j-=7; // j=j-7
		System.out.println("j="+j);
		
		int k=24;
		k/=2;
		System.out.println("k="+k);
		
		k%=5;
		System.out.println("k="+k);
		
		
		int x=3;
		x^=2;
		System.out.println("x="+x);

	}

}



