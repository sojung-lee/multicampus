package day09;
class SuperDemo{
	
	void sub() {
		System.out.println("SuperDemo's sub()");
	}
	final void bar() {
		System.out.println("SuperDemo's bar()");
	}
	
}/////////////////////
class SubDemo extends SuperDemo{
	//sub()오버라이딩 하기 
	@Override
	void sub() {
		System.out.println("sub() @@@@");
	}
	//final + method : 오버라이딩을 할 수 없다
	//bar()오버라이딩 하기
	//@Override
	//final void bar() {
		
	//}
}///////////////////////

public class FinalTest2 {

	public static void main(String[] args) {
		//SuperDemo의
		//sub(), bar()호출해보기
		SuperDemo sd=new SuperDemo();
		sd.sub();
		sd.bar();
		
		//SubDemo의 sub()호출해보기
		SubDemo sb=new SubDemo();
		sb.sub();
		sb.bar();
	}

}
