package day02;

public class PrimitiveType {
	
	public static void main(String[] args) {
		System.out.println("기본자료형---------------");
		System.out.println("");//Ctrl + D :현재 커서행을 삭제
		//Ctrl+S: 저장=>자동으로 컴파일됨(javac)
		//Ctrl+F11 : 현재 파일을 실행시킴(java)
		int a=10;
		int b=010;//정수(8진수) : 8진수는 앞에 0을 접두어로 붙인다 (0 ~ 7)
		System.out.println("a="+a);
		System.out.println("b="+b);
		int c=0123;//1*8^2 + 2*8^1 + 3*8^0=83
		System.out.println("c="+c);
		
		int d=0x12;//정수(16진수) :16진수는 앞에 0x를 접두어로 붙인다
		//1*16 + 2=>18
		// 0 ~ 15: 0~9 A,B,C,D,E,F
		int e=0x2ac;
		//2*16^2 + 10*16 + 12=684
		System.out.println("d="+d);
		System.out.println("e="+e);
		
		char c1='a';
		System.out.println("c1="+c1);
		char c2='0';
		System.out.println("c2="+c2);
		//int 미만의 자료형 (byte, short, char) =>연산에 사용되면 int로 자동형변환이 일어난다
		char c3=(char)(c2+1);//97+48
		System.out.println(c3);
		
		double d2=.456E+2;//0.456* 10^2=45.6
		System.out.println(d2);
	}

}
