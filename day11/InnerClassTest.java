package day11;

public class InnerClassTest {

	public static void main(String[] args) {
		//1) Outer클래스의 a,b변수값을 출력하세요
		Outer o1=new Outer();
		System.out.println("o1.a="+o1.a);
		System.out.println("Outer.b="+Outer.b);
		
		//2) Inner클래스의 c변수값 출력하고
		//			sub()메소드 호출하기
		Outer o2=new Outer();
		Outer.Inner oi=o2.new Inner();
		System.out.println("oi.c="+oi.c);
		oi.sub();
		
		
		Outer.Inner oi2=new Outer().new Inner();
		oi2.sub();
		
		//3) SInner클래스의 d, e 출력
		//			foo(), bar()호출하기
		Outer.SInner os=new Outer.SInner();
		System.out.println("os.d="+os.d);
		os.foo();
		
		System.out.println("Outer.SInner.e="+Outer.SInner.e);
		Outer.SInner.bar();
		
	}

}
