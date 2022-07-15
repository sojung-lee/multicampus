package day09;
interface Inter1{
	void a();
}////////////////////

interface Inter2{
	String b();
}//////////////////////

abstract class AbsClass{
	abstract void c();
}////////////////////////////
//인터페이스가 인터페이스를 상속받을 때는 extends로 상속받고, 다중 상속도 가능하다
interface HisInter extends Inter1, Inter2{
	String STR="Hello";
}/////////////////////////////////////////
//[1] MyClass2가 AbsClass와 HisInter를 상속받도록 하세요
//	  에러 없게 처리하세요

public class MyClass2 extends AbsClass implements HisInter {
	
	void c() {
		System.out.println("c()@@@");
	}
	public void a() {
		System.out.println("a()###");
	}
	public String b() {
		return "b()$$$$";
	}

	public static void main(String[] args) {
		//[2] MyClass2타입의 객체 생성해서 메서드 각각 호출하고 STR값도 출력해보기
		MyClass2 mc=new MyClass2();
		mc.c();
		mc.a();
		System.out.println(mc.b());
		System.out.println("-------------------");
		
		//[3] HisInter타입의 변수 선언하고 MyClass2객체 생성해서  메서드 각각 호출하고 STR값도 출력해보기
		HisInter hi=new MyClass2();
		hi.a();
		System.out.println(hi.b());
		System.out.println(HisInter.STR);
		System.out.println(MyClass2.STR);
		System.out.println("-------------------");
		//hi.c();[x]
		//[4] AbsClass타입의 변수 선언하고 MyClass2객체 생성해서  메서드 각각 호출하고 STR값도 출력해보기
		AbsClass ac =new MyClass2();
		ac.c();
		((MyClass2)ac).a();//
		//ac.b();//[x]

	}

}
