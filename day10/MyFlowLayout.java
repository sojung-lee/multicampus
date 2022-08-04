package day10;
import javax.swing.*;
import java.awt.*;

/* LayoutManger
 *  -[1] FlowLayout
 *  -[2] BorderLayout : 동,서,남,북,중앙 영역을 지정해서 배치
 *  -[3] GridLayout : 행과 열의 형태로 균등하게 배치
 * 레이아웃을 변경: setLayout()이용
 * */
public class MyFlowLayout extends JFrame {
	JButton b1,b2,b3,b4;
	JPanel p;//JPanel: 기본레이아웃은 FlowLayout

	public MyFlowLayout() {
		super("::MyFlowLayout::");
		p=new JPanel();
		this.add(p);
		p.setBackground(Color.white);
		//p.setLayout(new BorderLayout());
		//p.setLayout(new GridLayout(2,2));//2행2열
		p.setLayout(new FlowLayout(FlowLayout.RIGHT));
		
		b1=new JButton("     b1     ");
		b2=new JButton("b2");
		b3=new JButton("b3");
		b4=new JButton("b4");
		
		p.add(b1);
		p.add(b2);
		p.add(b3);
		p.add(b4);
		
		//창닫기
		this.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
	}

	public static void main(String[] args) {
		MyFlowLayout my=new MyFlowLayout();
		my.setSize(500,500);
		my.setVisible(true);
	}

}
