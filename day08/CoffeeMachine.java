package day08;

public class CoffeeMachine {
	int coffee, sugar, cream;
	Yuja yj;
	
	/*메소드 오버로딩(다중정의)
	 * - 메소드 이름은 동일하게
	 * - 매개변수의 자료형이 다르거나, 개수가 다르거나, 순서가 다르게
	 * - 반환타입 같아도 되고 달라도 된다
	 * */
	//밀크커피, 블랙커피, 크림커피, 설탕커피 만들다,....> 메소드
	public void makeTea(int coffee) {
		this.coffee = coffee;
		System.out.println("블랙 커피 나가요~~ 농도: "+this.coffee);
	}
	public void makeTea(int coffee, int sugar) {
		this.coffee=coffee;
		this.sugar=sugar;
		int n=this.coffee+this.sugar;
		System.out.println("설탕 커피 나가요~~ 농도: "+n);
	}
	public String makeTea(int coffee, int sugar, int cream) {
		this.coffee=coffee;
		this.sugar=sugar;
		this.cream=cream;
		String s="밀크 커피 나가요~~ 농도: "+(this.coffee+this.sugar+this.cream);
		return s;
	}
	//크림커피 메소드 구성해보기
	public void makeTea(int coffee, short cream) {
		this.coffee =coffee;
		this.cream = cream;
		System.out.println("크림 커피 나가요 ~ 농도: "+(this.coffee+this.cream));
	}
	public int makeTea(short cream, int coffee) {
		this.makeTea(coffee, cream); //this.메소드() : 자기자신의 메소드를 호출할 때 사용
		return (this.coffee+this.cream);
	}
	//유자차 만드는 메소드
	public void makeTea(Yuja y) {
		this.yj =y;
		System.out.println("[신메뉴] 유자차 나가요~~");
		System.out.println("유자: "+yj.getYuja());
		System.out.println("설탕: "+yj.getSugar());		
	}

	

}///////////////////////////////////





