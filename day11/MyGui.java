package day11;
import javax.swing.*;
import java.awt.*;
import java.awt.event.*;
/* 이벤트 처리
 *  [1] 이벤트 소스
 *  [2] 이벤트
 *  [3] 이벤트 핸들러
 * */

public class MyGui extends JFrame {
	JPanel p, pN;
	JTextArea ta;
	JButton bR, bG, bB, bExit;
	public MyGui() {
		super("::MyGui::");
		p=new JPanel();
		p.setBackground(Color.LIGHT_GRAY);
		add(p,"Center");
		
		p.setLayout(new BorderLayout());
		pN=new JPanel();//FlowLayout
		pN.setBackground(new Color(180, 50, 100));//r,g,b
		p.add(pN,"North");
		
		ta=new JTextArea("안녕");
		p.add(new JScrollPane(ta), "Center");
		
		bR=new JButton("Red");
		bG=new JButton("Green");
		bB=new JButton("Blue");
		bExit=new JButton("Exit");
		
		pN.add(bR);
		pN.add(bG);
		pN.add(bB);
		pN.add(bExit);
		//생성자 안에서 이벤트소스와 이벤트 핸들러를 연결해줘야 함 
		// - "이벤트소스.addXXXListener(이벤트핸들러객체)"
		MyEventHandler handler=new MyEventHandler();
		//MyGui.MyEventHandler handler2=this.new MyEventHandler();
		
		bR.addActionListener(handler);
		bG.addActionListener(handler);
		bB.addActionListener(handler);
		bExit.addActionListener(handler);
		
		
		this.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
	}//생성자-------------------------
	//이벤트 핸들러들은
	//XXXListener상속받아야 함
	class MyEventHandler implements ActionListener
	{
		@Override
		public void actionPerformed(ActionEvent e) {
			//이벤트가 발생된 객체의 참조값을 얻어보자.
			Object obj=e.getSource();//이벤트 소스에 대한 정보를 반환
			if(obj==bR) {
				//ta.setBackground(Color.red);
				ta.setForeground(Color.red);
				//이벤트 처리 코드
			}else if(obj==bG) {
				ta.setForeground(Color.green);
			}else if(obj==bB) {
				ta.setForeground(Color.blue);
			}else if(obj==bExit) {
				System.exit(0);//시스템 종료
			}
		}
		
	}/////////////////////
	

	public static void main(String[] args) {
		MyGui my=new MyGui();
		my.setSize(500,500);
		my.setVisible(true);

	}

}
