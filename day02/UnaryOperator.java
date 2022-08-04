package day02;

public class UnaryOperator {
	
	//main
	public static void main(String[] args) {
		System.out.println("1. 단항 연산자 (부호연산자: +, -)");
		int a=-8;
		System.out.println(+a);
		System.out.println(-a);
		System.out.println("2. 증감연산자: ++, --");
		int b=5;
		long c=9;
		b++;// b=b+1;
		c--;// c=c-1;
		System.out.println(b);//6
		System.out.println(c);//8
		++b;
		--c;
		System.out.println(b);//7
		System.out.println(c);//7
		//++,--가 단독으로 사용될 때는 변수 앞에오나 뒤에오나 별 차이는 없다
		//하지만 다른 수식과 함께 사용되면 ++,-- 가 앞에 오는 것과 뒤에 오는 것은 차이가 있다.
		float d=5.3f;
		double e=7.3;
		System.out.println(++d);//6.3
		System.out.println("d: "+d);//6.3
		
		System.out.println(e--);//7.3
		System.out.println("e: "+e);//6.3
		
		int x=10;
		int y=x++;
		System.out.println("x="+x);//11
		System.out.println("y="+y);//10
		
		x=10;
		int z=++x;
		System.out.println("x="+x);//11
		System.out.println("z="+z);//11
		
		System.out.println("3.비트별 not연산자: ~");
		//정수값을 비트로 바꾸고 연산을 수행하는데, 비트별 not연산자는 1은 0으로 바꾸고,
		//0은 1로 바꾸는 연산자
		// 3: 00000000 00000000 00000000 00000011
		//~3: 11111111 11111111 11111111 11111100
		// -1*2^31+(1*2^30) +(1*2^29) ..... = -4
		int m=3;
		System.out.println(m);
		System.out.println(~m);
		
		//문제1] 
		int n=0xfffffff1;
		//1111 1111 1111 1111 1111 1111 1111 0001
		//0000 0000 0000 0000 0000 0000 0000 1110
		System.out.println("~n: "+(~n));
		
		//42 :00000000 00000000 00000000 00101010
		//~42:11111111 11111111 11111111 11010101 ==>-43
		System.out.println(~42);
		
		System.out.println("4. 논리 부정 연산자: !");
		//boolean형에만 사용가능하다. int형에는 사용할 수 없다.
		//true => false로, false=>true로 바꾼다
		boolean bo=true;
		System.out.println(bo);//true
		System.out.println(!bo);//false
		System.out.println(!(3<2));
		
		int fo=5;
		System.out.println(fo==5.0); // == (등가연산자) : 기본자료형에서는 값이 같은지 비교한다
		System.out.println(fo!=5.0);// != : 기본자료형에서는 값이 다르면 true, 같으면 false
	}
}




